#include <streaming/webrtc_publisher.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/sdp/gstsdpmessage.h>
#include <gst/webrtc/webrtc.h>

namespace veilsight {
    namespace {
        bool ensure_gst_initialized() {
            if (gst_is_initialized()) return true;
            int argc = 0;
            char** argv = nullptr;
            gst_init(&argc, &argv);
            return gst_is_initialized();
        }

        bool gst_factory_available(const std::string& name) {
            if (!ensure_gst_initialized()) return false;
            GstElementFactory* factory = gst_element_factory_find(name.c_str());
            if (!factory) return false;
            gst_object_unref(factory);
            return true;
        }

        bool contains_stream(const std::vector<std::string>& streams, const std::string& stream_key) {
            return std::find(streams.begin(), streams.end(), stream_key) != streams.end();
        }

        std::string encoder_fragment(const std::string& encoder) {
            return encoder + " name=enc";
        }

        std::string h264_byte_stream_caps_fragment() {
            return "video/x-h264,stream-format=byte-stream,alignment=au";
        }

        std::string h264_rtp_caps_fragment(int payload_type) {
            return "application/x-rtp,media=(string)video,encoding-name=(string)H264,"
                   "payload=(int)" + std::to_string(payload_type) + ",packetization-mode=(string)1";
        }

        void set_property_if_present(GObject* object, const char* name, int value) {
            if (!object || !g_object_class_find_property(G_OBJECT_GET_CLASS(object), name)) return;
            g_object_set(object, name, value, nullptr);
        }

        bool promise_wait_ok(GstPromise* promise) {
            if (!promise) return false;
            return gst_promise_wait(promise) == GST_PROMISE_RESULT_REPLIED;
        }

        bool equals_ci(const std::string& a, const char* b) {
            const std::string rhs(b);
            if (a.size() != rhs.size()) return false;
            for (size_t i = 0; i < a.size(); ++i) {
                if (std::tolower(static_cast<unsigned char>(a[i])) !=
                    std::tolower(static_cast<unsigned char>(rhs[i]))) {
                    return false;
                }
            }
            return true;
        }

        bool starts_with_ci(const std::string& value, const char* prefix) {
            const std::string p(prefix);
            if (value.size() < p.size()) return false;
            for (size_t i = 0; i < p.size(); ++i) {
                if (std::tolower(static_cast<unsigned char>(value[i])) !=
                    std::tolower(static_cast<unsigned char>(p[i]))) {
                    return false;
                }
            }
            return true;
        }

        int parse_payload_prefix(const char* value) {
            if (!value) return -1;
            char* end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end == value || parsed < 0 || parsed > 127) return -1;
            return static_cast<int>(parsed);
        }

        bool fmtp_contains(const std::string& fmtp, const char* token) {
            return fmtp.find(token) != std::string::npos;
        }

        std::string fmtp_value(const std::string& fmtp, const char* key) {
            const std::string needle = std::string(key) + "=";
            const size_t start = fmtp.find(needle);
            if (start == std::string::npos) return {};
            const size_t value_start = start + needle.size();
            const size_t end = fmtp.find(';', value_start);
            return fmtp.substr(value_start, end == std::string::npos ? std::string::npos : end - value_start);
        }

        int select_h264_payload_type(const std::string& offer_sdp) {
            GstSDPMessage* sdp = nullptr;
            if (gst_sdp_message_new(&sdp) != GST_SDP_OK) return -1;
            const GstSDPResult parse_result =
                gst_sdp_message_parse_buffer(reinterpret_cast<const guint8*>(offer_sdp.data()),
                                             offer_sdp.size(),
                                             sdp);
            if (parse_result != GST_SDP_OK) {
                gst_sdp_message_free(sdp);
                return -1;
            }

            struct Candidate {
                int payload = -1;
                std::string fmtp;
            };
            std::vector<Candidate> candidates;

            for (guint media_i = 0; media_i < gst_sdp_message_medias_len(sdp); ++media_i) {
                const GstSDPMedia* media = gst_sdp_message_get_media(sdp, media_i);
                if (!media || !equals_ci(gst_sdp_media_get_media(media), "video")) continue;

                std::vector<int> h264_payloads;
                for (guint attr_i = 0; attr_i < gst_sdp_media_attributes_len(media); ++attr_i) {
                    const GstSDPAttribute* attr = gst_sdp_media_get_attribute(media, attr_i);
                    if (!attr || !attr->key || !attr->value) continue;
                    if (std::strcmp(attr->key, "rtpmap") == 0) {
                        const int payload = parse_payload_prefix(attr->value);
                        const char* space = std::strchr(attr->value, ' ');
                        if (payload >= 0 && space && starts_with_ci(space + 1, "H264/")) {
                            h264_payloads.push_back(payload);
                        }
                    }
                }

                for (const int payload : h264_payloads) {
                    std::string fmtp;
                    const std::string payload_prefix = std::to_string(payload) + " ";
                    for (guint attr_i = 0; attr_i < gst_sdp_media_attributes_len(media); ++attr_i) {
                        const GstSDPAttribute* attr = gst_sdp_media_get_attribute(media, attr_i);
                        if (!attr || !attr->key || !attr->value) continue;
                        if (std::strcmp(attr->key, "fmtp") == 0 &&
                            std::string(attr->value).rfind(payload_prefix, 0) == 0) {
                            fmtp = attr->value + payload_prefix.size();
                            break;
                        }
                    }
                    candidates.push_back({payload, std::move(fmtp)});
                }
                break;
            }

            gst_sdp_message_free(sdp);

            auto packetization_mode_1 = [](const Candidate& candidate) {
                return candidate.fmtp.empty() || fmtp_contains(candidate.fmtp, "packetization-mode=1");
            };
            auto profile = [](const Candidate& candidate) {
                return fmtp_value(candidate.fmtp, "profile-level-id");
            };

            for (const auto& candidate : candidates) {
                if (packetization_mode_1(candidate) && starts_with_ci(profile(candidate), "42e0")) {
                    return candidate.payload;
                }
            }
            for (const auto& candidate : candidates) {
                if (packetization_mode_1(candidate) && starts_with_ci(profile(candidate), "4200")) {
                    return candidate.payload;
                }
            }
            for (const auto& candidate : candidates) {
                if (packetization_mode_1(candidate)) return candidate.payload;
            }
            return candidates.empty() ? -1 : candidates.front().payload;
        }
    }

    struct WebRTCPublisher::Session {
        std::string id;
        std::string stream_key;
        StreamingConfig config;
        H264EncoderSelection encoder;
        int rtp_payload_type = 96;

        GstElement* pipeline = nullptr;
        GstElement* appsrc = nullptr;
        GstElement* webrtc = nullptr;

        mutable std::mutex mutex;
        std::condition_variable ice_cv;
        GstWebRTCICEGatheringState ice_state = GST_WEBRTC_ICE_GATHERING_STATE_NEW;
        GstWebRTCPeerConnectionState connection_state = GST_WEBRTC_PEER_CONNECTION_STATE_NEW;
        std::chrono::steady_clock::time_point last_seen = std::chrono::steady_clock::now();
        bool caps_set = false;
        int width = 0;
        int height = 0;
        int initial_width = 0;
        int initial_height = 0;
        uint64_t frame_index = 0;
        std::chrono::steady_clock::time_point first_frame_time;
        std::chrono::steady_clock::time_point previous_frame_time;

        Session(std::string session_id,
                std::string key,
                StreamingConfig cfg,
                H264EncoderSelection enc,
                int payload_type,
                int initial_w,
                int initial_h)
            : id(std::move(session_id)),
              stream_key(std::move(key)),
              config(std::move(cfg)),
              encoder(std::move(enc)),
              rtp_payload_type(payload_type),
              initial_width(initial_w),
              initial_height(initial_h) {}

        ~Session() {
            stop();
        }

        bool start(std::string* error) {
            if (!ensure_gst_initialized()) {
                if (error) *error = "failed to initialize GStreamer";
                return false;
            }

            const std::string pipeline_desc =
                "appsrc name=src is-live=true format=time do-timestamp=true block=false "
                "! queue leaky=downstream max-size-buffers=1 "
                "! videoconvert "
                "! video/x-raw,format=I420 "
                "! " + encoder_fragment(encoder.encoder) + " "
                "! h264parse config-interval=-1 "
                "! " + h264_byte_stream_caps_fragment() + " "
                "! rtph264pay pt=" + std::to_string(rtp_payload_type) + " config-interval=-1 aggregate-mode=zero-latency "
                "! " + h264_rtp_caps_fragment(rtp_payload_type) + " "
                "! webrtcbin name=webrtc bundle-policy=max-bundle";

            GError* gst_error = nullptr;
            pipeline = gst_parse_launch(pipeline_desc.c_str(), &gst_error);
            if (!pipeline) {
                if (error) {
                    *error = gst_error ? gst_error->message : "failed to create WebRTC pipeline";
                }
                if (gst_error) g_error_free(gst_error);
                return false;
            }
            if (gst_error) g_error_free(gst_error);

            appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "src");
            webrtc = gst_bin_get_by_name(GST_BIN(pipeline), "webrtc");
            GstElement* enc = gst_bin_get_by_name(GST_BIN(pipeline), "enc");

            if (!appsrc || !webrtc) {
                if (error) *error = "WebRTC pipeline missing appsrc or webrtcbin";
                if (enc) gst_object_unref(enc);
                stop();
                return false;
            }

            g_object_set(appsrc,
                         "stream-type", GST_APP_STREAM_TYPE_STREAM,
                         "format", GST_FORMAT_TIME,
                         "is-live", TRUE,
                         "do-timestamp", TRUE,
                         "block", FALSE,
                         nullptr);
            if (initial_width > 0 && initial_height > 0) {
                set_caps(initial_width, initial_height);
            }

            if (enc) {
                if (encoder.encoder == "x264enc") {
                    set_property_if_present(G_OBJECT(enc), "tune", 0x00000004); // zerolatency
                    set_property_if_present(G_OBJECT(enc), "speed-preset", 1);  // ultrafast
                    set_property_if_present(G_OBJECT(enc), "key-int-max", config.keyframe_interval_frames);
                    set_property_if_present(G_OBJECT(enc), "bitrate", config.bitrate_kbps);
                    set_property_if_present(G_OBJECT(enc), "byte-stream", 1);
                    set_property_if_present(G_OBJECT(enc), "bframes", 0);
                    set_property_if_present(G_OBJECT(enc), "cabac", 0);
                    set_property_if_present(G_OBJECT(enc), "dct8x8", 0);
                } else {
                    set_property_if_present(G_OBJECT(enc), "bitrate", config.bitrate_kbps);
                    set_property_if_present(G_OBJECT(enc), "keyframe-period", config.keyframe_interval_frames);
                    set_property_if_present(G_OBJECT(enc), "gop-size", config.keyframe_interval_frames);
                }
                gst_object_unref(enc);
            }

            if (!config.webrtc.stun_servers.empty() &&
                g_object_class_find_property(G_OBJECT_GET_CLASS(webrtc), "stun-server")) {
                g_object_set(webrtc, "stun-server", config.webrtc.stun_servers.front().c_str(), nullptr);
            }

            g_signal_connect(webrtc, "notify::ice-gathering-state", G_CALLBACK(+[](
                                 GObject* object, GParamSpec*, gpointer user_data) {
                auto* session = static_cast<Session*>(user_data);
                GstWebRTCICEGatheringState state = GST_WEBRTC_ICE_GATHERING_STATE_NEW;
                g_object_get(object, "ice-gathering-state", &state, nullptr);
                {
                    std::lock_guard lk(session->mutex);
                    session->ice_state = state;
                }
                session->ice_cv.notify_all();
            }), this);

            g_signal_connect(webrtc, "notify::connection-state", G_CALLBACK(+[](
                                 GObject* object, GParamSpec*, gpointer user_data) {
                auto* session = static_cast<Session*>(user_data);
                GstWebRTCPeerConnectionState state = GST_WEBRTC_PEER_CONNECTION_STATE_NEW;
                g_object_get(object, "connection-state", &state, nullptr);
                std::lock_guard lk(session->mutex);
                session->connection_state = state;
            }), this);

            const GstStateChangeReturn state_result = gst_element_set_state(pipeline, GST_STATE_PLAYING);
            if (state_result == GST_STATE_CHANGE_FAILURE) {
                if (error) *error = "failed to set WebRTC pipeline to PLAYING";
                stop();
                return false;
            }
            return true;
        }

        void stop() {
            GstElement* pipeline_to_stop = nullptr;
            GstElement* appsrc_to_unref = nullptr;
            GstElement* webrtc_to_unref = nullptr;
            {
                std::lock_guard lk(mutex);
                pipeline_to_stop = pipeline;
                appsrc_to_unref = appsrc;
                webrtc_to_unref = webrtc;
                pipeline = nullptr;
                appsrc = nullptr;
                webrtc = nullptr;
            }
            if (pipeline_to_stop) gst_element_set_state(pipeline_to_stop, GST_STATE_NULL);
            if (appsrc_to_unref) gst_object_unref(appsrc_to_unref);
            if (webrtc_to_unref) gst_object_unref(webrtc_to_unref);
            if (pipeline_to_stop) gst_object_unref(pipeline_to_stop);
        }

        bool set_remote_offer(const std::string& offer_sdp, std::string* error) {
            GstSDPMessage* sdp = nullptr;
            if (gst_sdp_message_new(&sdp) != GST_SDP_OK) {
                if (error) *error = "failed to allocate SDP message";
                return false;
            }
            const GstSDPResult parse_result =
                gst_sdp_message_parse_buffer(reinterpret_cast<const guint8*>(offer_sdp.data()),
                                             offer_sdp.size(),
                                             sdp);
            if (parse_result != GST_SDP_OK) {
                gst_sdp_message_free(sdp);
                if (error) *error = "invalid SDP offer";
                return false;
            }

            GstWebRTCSessionDescription* offer =
                gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_OFFER, sdp);
            GstPromise* promise = gst_promise_new();
            g_signal_emit_by_name(webrtc, "set-remote-description", offer, promise);
            const bool ok = promise_wait_ok(promise);
            gst_promise_unref(promise);
            gst_webrtc_session_description_free(offer);
            if (!ok && error) *error = "failed to set remote WebRTC offer";
            return ok;
        }

        bool create_answer(std::string* answer_sdp, std::string* error) {
            GstPromise* answer_promise = gst_promise_new();
            g_signal_emit_by_name(webrtc, "create-answer", nullptr, answer_promise);
            if (!promise_wait_ok(answer_promise)) {
                gst_promise_unref(answer_promise);
                if (error) *error = "failed to create WebRTC answer";
                return false;
            }

            const GstStructure* reply = gst_promise_get_reply(answer_promise);
            GstWebRTCSessionDescription* answer = nullptr;
            gst_structure_get(reply, "answer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &answer, nullptr);
            gst_promise_unref(answer_promise);
            if (!answer) {
                if (error) *error = "WebRTC answer was empty";
                return false;
            }

            GstPromise* local_promise = gst_promise_new();
            g_signal_emit_by_name(webrtc, "set-local-description", answer, local_promise);
            const bool local_ok = promise_wait_ok(local_promise);
            gst_promise_unref(local_promise);
            if (!local_ok) {
                gst_webrtc_session_description_free(answer);
                if (error) *error = "failed to set local WebRTC answer";
                return false;
            }

            wait_for_ice_gathering();

            GstWebRTCSessionDescription* local_description = nullptr;
            g_object_get(webrtc, "local-description", &local_description, nullptr);
            GstWebRTCSessionDescription* description_to_serialize =
                local_description ? local_description : answer;

            gchar* text = gst_sdp_message_as_text(description_to_serialize->sdp);
            if (text) {
                *answer_sdp = text;
                g_free(text);
            }
            if (local_description) gst_webrtc_session_description_free(local_description);
            gst_webrtc_session_description_free(answer);
            if (answer_sdp->empty()) {
                if (error) *error = "local SDP answer was empty";
                return false;
            }
            return true;
        }

        void wait_for_ice_gathering() {
            std::unique_lock lk(mutex);
            const auto timeout = std::chrono::milliseconds(config.webrtc.ice_gathering_timeout_ms);
            ice_cv.wait_for(lk, timeout, [&] {
                return ice_state == GST_WEBRTC_ICE_GATHERING_STATE_COMPLETE;
            });
        }

        bool should_expire(std::chrono::steady_clock::time_point now,
                           std::chrono::seconds timeout) const {
            std::lock_guard lk(mutex);
            return connection_state == GST_WEBRTC_PEER_CONNECTION_STATE_FAILED ||
                   connection_state == GST_WEBRTC_PEER_CONNECTION_STATE_CLOSED ||
                   connection_state == GST_WEBRTC_PEER_CONNECTION_STATE_DISCONNECTED ||
                   now - last_seen > timeout;
        }

        void set_caps(int new_width, int new_height) {
            width = new_width;
            height = new_height;
            GstCaps* caps = gst_caps_new_simple("video/x-raw",
                                                "format", G_TYPE_STRING, "BGR",
                                                "width", G_TYPE_INT, width,
                                                "height", G_TYPE_INT, height,
                                                "framerate", GST_TYPE_FRACTION, 0, 1,
                                                nullptr);
            gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
            gst_caps_unref(caps);
            caps_set = true;
        }

        void push_frame(const cv::Mat& frame) {
            std::unique_lock lk(mutex, std::try_to_lock);
            if (!lk.owns_lock() || !appsrc || frame.empty() || frame.type() != CV_8UC3) return;

            if (!caps_set || width != frame.cols || height != frame.rows) {
                set_caps(frame.cols, frame.rows);
            }

            const gsize bytes = static_cast<gsize>(frame.total() * frame.elemSize());
            GstBuffer* buffer = gst_buffer_new_allocate(nullptr, bytes, nullptr);
            if (!buffer) return;

            GstMapInfo map;
            if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
                gst_buffer_unref(buffer);
                return;
            }
            if (frame.isContinuous()) {
                std::memcpy(map.data, frame.data, bytes);
            } else {
                guint8* dst = map.data;
                const size_t row_bytes = static_cast<size_t>(frame.cols * frame.elemSize());
                for (int y = 0; y < frame.rows; ++y) {
                    std::memcpy(dst + y * row_bytes, frame.ptr(y), row_bytes);
                }
            }
            gst_buffer_unmap(buffer, &map);

            const auto now = std::chrono::steady_clock::now();
            if (frame_index == 0) {
                first_frame_time = now;
                previous_frame_time = now;
                GST_BUFFER_PTS(buffer) = 0;
                GST_BUFFER_DURATION(buffer) = GST_CLOCK_TIME_NONE;
            } else {
                const auto pts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - first_frame_time);
                const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - previous_frame_time);
                GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(pts_ns.count());
                GST_BUFFER_DURATION(buffer) = duration_ns.count() > 0
                    ? static_cast<GstClockTime>(duration_ns.count())
                    : GST_CLOCK_TIME_NONE;
                previous_frame_time = now;
            }
            ++frame_index;
            last_seen = std::chrono::steady_clock::now();

            const GstFlowReturn flow = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
            if (flow != GST_FLOW_OK && flow != GST_FLOW_FLUSHING) {
                thread_local bool logged = false;
                if (!logged) {
                    std::cerr << "[WebRTC] appsrc push failed for " << stream_key
                              << " flow=" << flow << "\n";
                    logged = true;
                }
            }
        }
    };

    H264EncoderSelector::H264EncoderSelector(AvailabilityFn availability)
        : availability_(std::move(availability)) {
        if (!availability_) availability_ = gst_factory_available;
    }

    H264EncoderSelection H264EncoderSelector::select(const StreamingConfig& config) const {
        if (!availability_("webrtcbin")) {
            return {false, {}, "GStreamer webrtcbin element is unavailable"};
        }

        if (config.encoder != "auto") {
            if (availability_(config.encoder)) return {true, config.encoder, {}};
            return {false, config.encoder, "configured H.264 encoder is unavailable: " + config.encoder};
        }

        static const std::vector<std::string> kEncoders = {
            "v4l2h264enc",
            "vaapih264enc",
            "nvh264enc",
            "x264enc",
            "openh264enc",
        };
        for (const auto& encoder : kEncoders) {
            if (availability_(encoder)) return {true, encoder, {}};
        }
        return {false, {}, "no supported H.264 encoder is available"};
    }

    WebRTCPublisher::WebRTCPublisher(StreamingConfig config)
        : config_(std::move(config)),
          encoder_(H264EncoderSelector().select(config_)) {
        if (available()) {
            std::cerr << "[WebRTC] using H.264 encoder: " << encoder_.encoder << "\n";
        } else {
            std::cerr << "[WebRTC] unavailable: " << unavailable_reason() << "\n";
        }
    }

    bool WebRTCPublisher::available() const {
        return config_.webrtc.enabled && encoder_.available;
    }

    std::string WebRTCPublisher::unavailable_reason() const {
        if (!config_.webrtc.enabled) return "WebRTC is disabled by config";
        return encoder_.error;
    }

    H264EncoderSelection WebRTCPublisher::encoder_selection() const {
        return encoder_;
    }

    void WebRTCPublisher::register_stream(const std::string& stream_key) {
        std::lock_guard lk(mutex_);
        if (!contains_stream(streams_, stream_key)) streams_.push_back(stream_key);
    }

    void WebRTCPublisher::publish_frame(const std::string& stream_key, const cv::Mat& frame) {
        std::vector<std::shared_ptr<Session>> targets;
        {
            std::unique_lock lk(mutex_, std::try_to_lock);
            if (!lk.owns_lock()) return;
            if (!frame.empty()) latest_dimensions_[stream_key] = {frame.cols, frame.rows};
            targets.reserve(sessions_.size());
            for (const auto& [_, session] : sessions_) {
                if (session && session->stream_key == stream_key) targets.push_back(session);
            }
        }
        for (auto& session : targets) {
            session->push_frame(frame);
        }
    }

    WebRTCPublisher::OfferResult WebRTCPublisher::handle_offer(const std::string& stream_key,
                                                               const std::string& sdp_offer) {
        expire_idle_sessions();

        if (!available()) {
            return {503, "text/plain", unavailable_reason(), {}};
        }
        if (sdp_offer.empty()) {
            return {400, "text/plain", "empty SDP offer", {}};
        }

        const std::string session_id = make_session_id_();
        const int rtp_payload_type = select_h264_payload_type(sdp_offer);
        if (rtp_payload_type < 0) {
            return {406, "text/plain", "SDP offer does not include a supported H.264 video payload", {}};
        }
        int initial_width = 0;
        int initial_height = 0;
        {
            std::lock_guard lk(mutex_);
            if (!contains_stream(streams_, stream_key)) {
                return {404, "text/plain", "unknown stream", {}};
            }

            size_t peers = 0;
            for (const auto& [_, existing] : sessions_) {
                if (existing && existing->stream_key == stream_key) ++peers;
            }
            if (peers >= static_cast<size_t>(config_.webrtc.max_peers_per_stream)) {
                return {429, "text/plain", "too many WebRTC peers for stream", {}};
            }
            auto dims = latest_dimensions_.find(stream_key);
            if (dims != latest_dimensions_.end()) {
                initial_width = dims->second.first;
                initial_height = dims->second.second;
            }
        }

        auto session = std::make_shared<Session>(session_id,
                                                 stream_key,
                                                 config_,
                                                 encoder_,
                                                 rtp_payload_type,
                                                 initial_width,
                                                 initial_height);
        {
            std::lock_guard lk(mutex_);
            sessions_[session_id] = session;
        }

        std::string error;
        std::string answer_sdp;
        if (!session->start(&error) ||
            !session->set_remote_offer(sdp_offer, &error) ||
            !session->create_answer(&answer_sdp, &error)) {
            delete_session(session_id);
            return {500, "text/plain", error.empty() ? "failed to create WebRTC session" : error, {}};
        }

        return {201, "application/sdp", answer_sdp, "/webrtc/whep/sessions/" + session_id};
    }

    bool WebRTCPublisher::delete_session(const std::string& session_id) {
        std::shared_ptr<Session> removed;
        {
            std::lock_guard lk(mutex_);
            auto it = sessions_.find(session_id);
            if (it == sessions_.end()) return false;
            removed = std::move(it->second);
            sessions_.erase(it);
        }
        if (removed) removed->stop();
        return true;
    }

    void WebRTCPublisher::clear_sessions() {
        std::map<std::string, std::shared_ptr<Session>> removed;
        {
            std::lock_guard lk(mutex_);
            removed.swap(sessions_);
        }
        for (auto& [_, session] : removed) {
            if (session) session->stop();
        }
    }

    void WebRTCPublisher::expire_idle_sessions() {
        const auto now = std::chrono::steady_clock::now();
        const auto timeout = std::chrono::seconds(config_.webrtc.session_idle_timeout_s);
        std::vector<std::string> expired;
        {
            std::lock_guard lk(mutex_);
            for (const auto& [id, session] : sessions_) {
                if (session && session->should_expire(now, timeout)) expired.push_back(id);
            }
        }
        for (const auto& id : expired) {
            delete_session(id);
        }
    }

    size_t WebRTCPublisher::active_sessions(const std::string& stream_key) const {
        std::lock_guard lk(mutex_);
        size_t peers = 0;
        for (const auto& [_, session] : sessions_) {
            if (session && session->stream_key == stream_key) ++peers;
        }
        return peers;
    }

    std::string WebRTCPublisher::make_session_id_() {
        static std::atomic<uint64_t> next_id{1};
        std::ostringstream oss;
        oss << "sess-" << next_id.fetch_add(1, std::memory_order_relaxed);
        return oss.str();
    }
}
