#include <encode/webrtc_manager.hpp>

#include <gst/app/gstappsrc.h>
#include <gst/webrtc/webrtc.h>

#include <iostream>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <unordered_map>

#include <common/config.hpp>

namespace ss {
    struct StreamCtx {
        GstElement* pipeline = nullptr;
        GstElement* appsrc = nullptr;
        GstElement* webrtc = nullptr;
        GstWebRTCDataChannel* data_channel = nullptr;

        int fps = 0, width = 0, height = 0;

        std::mutex sdp_mtx;
        std::condition_variable sdp_cv;
        std::string answer_sdp;
        bool answer_ready = false;

        ~StreamCtx() {
            if (data_channel) {
                g_object_unref(data_channel);
                data_channel = nullptr;
            }
            if (appsrc) {
                gst_object_unref(appsrc);
                appsrc = nullptr;
            }
            if (webrtc) {
                gst_object_unref(webrtc);
                webrtc = nullptr;
            }
            if (pipeline) {
                gst_element_set_state(pipeline, GST_STATE_NULL);
                gst_object_unref(pipeline);
                pipeline = nullptr;
            }
        }
    };

    struct WebRTCManager::Impl {
        std::unordered_map<std::string, std::unique_ptr<StreamCtx>> streams;
    };

    static void on_data_channel(GstElement*, /* webrtc */
                                GstWebRTCDataChannel* channel,
                                gpointer user_data) {
        auto* ctx = static_cast<StreamCtx*>(user_data);
        if (ctx->data_channel) {
            g_object_unref(ctx->data_channel);
        }
        ctx->data_channel = channel;
        if (channel) {
            g_object_ref(channel);
        }
    }

    static void on_answer_created(GstPromise* promise, gpointer user_data) {
        auto* ctx = static_cast<StreamCtx*>(user_data);

        const GstStructure* reply = gst_promise_get_reply(promise);
        GstWebRTCSessionDescription* answer = nullptr;

        gst_structure_get(reply, "answer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &answer, nullptr);
        GstPromise* p = gst_promise_new();
        g_signal_emit_by_name(ctx->webrtc, "set-local-description", answer, p);
        gst_promise_unref(p);

        gchar* sdp_str = gst_sdp_message_as_text(answer->sdp);

        {
            std::lock_guard lk(ctx->sdp_mtx);
            ctx->answer_sdp = sdp_str ? sdp_str : "";
            ctx->answer_ready = true;
        }
        ctx->sdp_cv.notify_all();

        if (sdp_str) g_free(sdp_str);
        gst_webrtc_session_description_free(answer);
        gst_promise_unref(promise);
    }

    WebRTCManager::WebRTCManager()
        : impl_(std::make_unique<Impl>()) {
        gst_init(nullptr, nullptr);
    }

    bool WebRTCManager::add_stream(const std::string& stream_id, const OutputConfig& cfg) {
        auto ctx = std::make_unique<StreamCtx>();
        ctx->fps = cfg.fps;
        ctx->width = cfg.width;
        ctx->height = cfg.height;

        if (ctx->fps <= 0 || ctx->width <= 0 || ctx->height <= 0) {
            std::cerr << "[WebRTC] Invalid configuration.";
            return false;
        }

        std::string pipeline_desc =
            "appsrc name=src is-live=true format=time do-timestamp=true "
            "! videoconvert ! videorate "
            "! video/x-raw,format=I420,framerate=" + std::to_string(ctx->fps) + "/1 "
            "! x264enc tune=zerolatency speed-preset=ultrafast bitrate=1500 key-int-max=30 "
            "! rtph264pay config-interval=1 pt=96 "
            "! webrtcbin name=webrtc";

        GError* err = nullptr;
        ctx->pipeline = gst_parse_launch(pipeline_desc.c_str(), &err);
        if (!ctx->pipeline) {
            std::cerr << "[WebRTC] Failed to create pipeline.";
            if (err) {
                std::cerr << ": " << err->message;
                g_error_free(err);
            }
            return false;
        }

        ctx->appsrc = gst_bin_get_by_name(GST_BIN(ctx->pipeline), "src");
        ctx->webrtc = gst_bin_get_by_name(GST_BIN(ctx->pipeline), "webrtc");

        if (!ctx->appsrc || !ctx->webrtc) {
            std::cerr << "[WebRTC] Missing appsrc/webrtcbin.";
            return false;
        }

        GstCaps* caps = gst_caps_new_simple(
            "video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, ctx->width,
            "height", G_TYPE_INT, ctx->height,
            "framerate", GST_TYPE_FRACTION, ctx->fps, 1,
            nullptr);
        gst_app_src_set_caps(GST_APP_SRC(ctx->appsrc), caps);
        gst_caps_unref(caps);

        g_signal_connect(ctx->webrtc, "on-data-channel", G_CALLBACK(on_data_channel), ctx.get());
        auto ret = gst_element_set_state(ctx->pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "[WebRTC]l faied to set pipeline to playing";
            return false;
        }

        impl_->streams.emplace(stream_id, std::move(ctx));
        return true;
    }

    void WebRTCManager::push_frame(const std::string& stream_id,
                                   const cv::Mat& frame,
                                   int64_t pts_ns) {
        auto it = impl_->streams.find(stream_id);
        if (it == impl_->streams.end() || frame.empty()) return;
        StreamCtx* ctx = it->second.get();

        const size_t size = frame.total() * frame.elemSize();
        GstBuffer* buf = gst_buffer_new_allocate(nullptr, size, nullptr);

        GstMapInfo map;
        gst_buffer_map(buf, &map, GST_MAP_WRITE);
        std::memcpy(map.data, frame.data, size);
        gst_buffer_unmap(buf, &map);

        GST_BUFFER_PTS(buf) = pts_ns;
        GST_BUFFER_DURATION(buf) = gst_util_uint64_scale_int(1, GST_SECOND, ctx->fps);

        GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(ctx->appsrc), buf);
        if (ret != GST_FLOW_OK) {
            std::cerr << "[WebRTC] Failed to push buffer to stream. ret: " << ret << "\n";
        }
    }

    void WebRTCManager::push_meta(const std::string& stream_id, const std::string& meta) {
        auto it = impl_->streams.find(stream_id);
        if (it == impl_->streams.end()) return;
        StreamCtx* ctx = it->second.get();

        if (!ctx->data_channel) return;
        gst_webrtc_data_channel_send_string(ctx->data_channel, meta.c_str());
    }

    std::string WebRTCManager::handle_offer(const std::string& stream_id, const std::string& sdp_offer) {
        auto it = impl_->streams.find(stream_id);
        if (it == impl_->streams.end()) return "";
        StreamCtx* ctx = it->second.get();

        GstSDPMessage* sdp = nullptr;
        gst_sdp_message_new(&sdp);
        if (gst_sdp_message_parse_buffer(
                reinterpret_cast<const guint8*>(sdp_offer.data()),
                sdp_offer.size(), sdp) != GST_SDP_OK) {
            std::cerr << "[WebRTC] Failed to parse SDP message.";
            gst_sdp_message_free(sdp);
            return "";
        }

        GstWebRTCSessionDescription* offer =
            gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_OFFER, sdp);

        GstPromise* p = gst_promise_new();
        g_signal_emit_by_name(ctx->webrtc, "set-remote-description", offer, p);
        gst_promise_unref(p);
        gst_webrtc_session_description_free(offer);

        {
            std::lock_guard lk(ctx->sdp_mtx);
            ctx->answer_ready = false;
            ctx->answer_sdp.clear();
        }

        GstPromise* ans_promise =
            gst_promise_new_with_change_func(on_answer_created, ctx, nullptr);
        g_signal_emit_by_name(ctx->webrtc, "create-answer", nullptr, ans_promise);

        std::unique_lock lk(ctx->sdp_mtx);
        bool success = ctx->sdp_cv.wait_for(lk, std::chrono::seconds(5), [&]{ return ctx->answer_ready; });
        if (!success) {
            std::cerr << "[WebRTC] Timeout waiting for the answer.";
            return "";
        }
        return ctx->answer_sdp;
    }
}
