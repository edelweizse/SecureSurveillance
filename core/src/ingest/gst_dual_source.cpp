#include <ingest/gst_dual_source.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include <iostream>
#include <mutex>
#include <utility>

namespace ss {
    constexpr int64_t PTS_TOLERANCE_NS = 1000000;

    GstDualSource::GstDualSource(std::string pipeline,
                                 std::string id,
                                 std::string sink_inf_name,
                                 std::string sink_ui_name)
        : pipeline_str_(std::move(pipeline)),
          id_(std::move(id)),
          sink_inf_name_(std::move(sink_inf_name)),
          sink_ui_name_(std::move(sink_ui_name)) {}

    bool GstDualSource::start() {
        static std::once_flag gst_init_flag;
        std::call_once(gst_init_flag, []{ gst_init(nullptr, nullptr); });

        GError *err = nullptr;
        pipeline_ = gst_parse_launch(pipeline_str_.c_str(), &err);
        if (!pipeline_) {
            if (err) {
                std::cerr << "[GStreamer](start) parse_launch error: " << err->message << "\n";
                g_error_free(err);
            } else {
                std::cerr << "[GStreamer](start) parse_launch failed (unk error)\n";
            }
            return false;
        }

        sink_inf_ = gst_bin_get_by_name(GST_BIN(pipeline_), sink_inf_name_.c_str());
        sink_ui_ = gst_bin_get_by_name(GST_BIN(pipeline_), sink_ui_name_.c_str());

        if (!sink_inf_ || !sink_ui_) {
            std::cerr << "[GStreamer](start) missing appsink(s): "
                      << sink_inf_name_ << "and/or" << sink_ui_name_ << "\n";
            stop();
            return false;
        }

        for (auto* s : {sink_inf_, sink_ui_}) {
            auto* appsink = GST_APP_SINK(s);
            gst_app_sink_set_drop(appsink, TRUE);
            gst_app_sink_set_max_buffers(appsink, 1);
            gst_app_sink_set_emit_signals(appsink, FALSE);
        }

        auto ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "[GStreamer](start) Failed to set pipeline to PLAYING.\n";
            stop();
            return false;
        }

        return true;
    }



    bool GstDualSource::pull_frame_(GstElement* sink,
                                      cv::Mat& out,
                                      int64_t& pts_ns,
                                      int timeout_ms) {
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink),timeout_ms * GST_MSECOND);
        if (!sample) return false;

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstCaps* caps = gst_sample_get_caps(sample);
        if (!buffer || !caps) {
            gst_sample_unref(sample);
            // std::cerr << "[GStreamer](pull_raw_bgr) Failed to get buffer and/or caps.\n";
            return false;
        }

        GstStructure* st = gst_caps_get_structure(caps, 0);
        int width = 0, height = 0;
        gst_structure_get_int(st, "width", &width);
        gst_structure_get_int(st, "height", &height);
        if (width <= 0 || height <= 0) {
            gst_sample_unref(sample);
            // std::cerr << "[GStreamer] (pull_raw_bgr) Failed to pull Width and/or Height from caps structure.\n";
            return false;
        }

        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ) || !map.data || map.size == 0) {
            gst_sample_unref(sample);
            // std::cerr << "[GStreamer](pull_raw_bgr) Failed to get buffer map.\n";
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
            // std::cerr << "[GStreamer](pull_raw_bgr) Failed to get buffer map.\n";
            return false;
        }

        cv::Mat tmp(height, width, CV_8UC3, map.data, stride);
        out = tmp.clone();
        pts_ns = (buffer->pts == GST_CLOCK_TIME_NONE)
                   ? 0
                   : static_cast<int64_t>(buffer->pts);

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);

        return true;
    }

    bool GstDualSource::read_frame(std::shared_ptr<FrameBundle>& out, int timeout_ms) {
        if (!sink_ui_ || !sink_inf_) {
            std::cerr << "[GStreamer](read_ui) Failed to get sink_ui/sink_inf instance.\n";
            return false;
        }
        cv::Mat inf, ui;
        int64_t pts_inf = 0, pts_ui = 0;
        if (!pull_frame_(sink_inf_, inf, pts_inf, timeout_ms)) {
            // std::cerr << "[GStreamer](read_frame) Failed to pull from inf sink.\n)";
            return false;
        }

        if (!pull_frame_(sink_ui_, ui, pts_ui, timeout_ms)) {
            // std::cerr << "[GStreamer](read_frame) Failed to pull from ui sink.\n)";
            return false;
        }
        int64_t pts_diff = std::abs(pts_inf - pts_ui);
        if (pts_diff > PTS_TOLERANCE_NS) {
            std::cerr << "[GStreamer](read_frame) PTS Mismatch is more tolerance";
            return false;
        }

        auto bundle = std::make_shared<FrameBundle>();
        bundle->stream_id = id_;
        bundle->frame_id = frame_id_++;
        bundle->pts_ns = pts_inf;

        bundle->inf_bgr = std::make_shared<const cv::Mat>(std::move(inf));
        bundle->ui_bgr = std::make_shared<cv::Mat>(std::move(ui));

        if (bundle->inf_bgr->cols > 0 && bundle->inf_bgr->rows > 0) {
            bundle->sx = float(bundle->ui_bgr->cols) / float(bundle->inf_bgr->cols);
            bundle->sy = float(bundle->ui_bgr->rows) / float(bundle->inf_bgr->rows);
        } else {
            std::cerr << "[GStreamer](read_frame) Warning: infr streams has zero dims";
            bundle->sx = 1.0f;
            bundle->sy = 1.0f;
        }

        out = std::move(bundle);
        return true;
    }

    void GstDualSource::stop() {
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);

            if (sink_inf_) { gst_object_unref(sink_inf_); sink_inf_ = nullptr; }
            if (sink_ui_) { gst_object_unref(sink_ui_); sink_ui_ = nullptr; }

            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

    GstDualSource::~GstDualSource() { stop(); }
}
