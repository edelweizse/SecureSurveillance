#include <ingest/gst_frame_source.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <opencv2/core.hpp>
#include <iostream>

namespace ss {
    bool GstFrameSource::gst_inited_ = false;

    GstFrameSource::GstFrameSource(std::string pipeline, std::string id, std::string sink_name)
        : pipeline_str_(std::move(pipeline)), id_(std::move(id)), sink_name_(std::move(sink_name)) {}

    bool GstFrameSource::start() {
        static std::once_flag gst_init_flag;
        std::call_once(gst_init_flag, [] { gst_init(nullptr, nullptr); });

        GError* err = nullptr;
        pipeline_ = gst_parse_launch(pipeline_str_.c_str(), &err);
        if (!pipeline_) {
            if (err) {
                std::cerr << "[GStreamer] parse_launch error: " << err->message << "\n";
                g_error_free(err);
            } else {
                std::cerr << "[GStreamer] parse_launch failed (unk error)\n";
            }
            return false;
        }

        sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), sink_name_.c_str());
        if (!sink_) {
            std::cerr << "[GStreamer] appsink named " << sink_name_ <<" not found.\n";
            stop();
            return false;
        }

        GstAppSink* appsink = GST_APP_SINK(sink_);
        gst_app_sink_set_drop(appsink, TRUE);
        gst_app_sink_set_max_buffers(appsink, 2);
        gst_app_sink_set_emit_signals(appsink, FALSE);

        GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "[GStreamer] Failed to set pipeline to PLAYING\n";
            stop();
            return false;
        }

        return true;
    }

    bool GstFrameSource::read(FramePacket& out, int timeout_ms) {
        if (!sink_) return false;

        GstSample* sample = gst_app_sink_try_pull_sample(
            GST_APP_SINK(sink_), timeout_ms * GST_MSECOND);

        if (!sample) return false;

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);
        if (!buffer || !caps) {
            gst_sample_unref(sample);
            return false;
        }

        GstStructure* st = gst_caps_get_structure(caps, 0);
        int width = 0, height = 0;
        gst_structure_get_int(st, "width", &width);
        gst_structure_get_int(st, "height", &height);
        if (width <= 0 || height <= 0) {
            gst_sample_unref(sample);
            return false;
        }

        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ) || !map.data || map.size == 0) {
            gst_sample_unref(sample);
            return false;
        }

        GstVideoInfo vinfo;
        int stride = width * 3;
        if (gst_video_info_from_caps(&vinfo, caps)) {
            int s0 = GST_VIDEO_INFO_PLANE_STRIDE(&vinfo, 0);
            if (s0 > 0) stride = s0;
        }

        const size_t min_bytes = static_cast<size_t>(stride) * static_cast<size_t>(height);
        if (map.size < min_bytes) {
            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            return false;
        }

        cv::Mat tmp(height, width, CV_8UC3, (void*)map.data, stride);
        out.bgr = tmp.clone();

        out.pts_ns = (buffer->pts == GST_CLOCK_TIME_NONE) ? 0 : static_cast<int64_t>(buffer->pts);
        out.frame_id = frame_id_++;

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return true;
    }

    void GstFrameSource::stop() {
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);

            if (sink_) {
                gst_object_unref(sink_);
                sink_ = nullptr;
            }
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

    GstFrameSource::~GstFrameSource() {
        stop();
    }
}
