#pragma once
#include <string>
#include <memory>
#include <opencv2/core.hpp>

struct _GstElement;
using GstElement = _GstElement;

namespace ss {
    struct FrameBundle {
        std::string stream_id;

        int64_t frame_id = 0;
        int64_t pts_ns = 0;

        std::shared_ptr<const cv::Mat> inf_bgr;
        std::shared_ptr<cv::Mat> ui_bgr;

        float sx = 1.f;  // uix = infx * sx
        float sy = 1.f;  // uiy = infy * sy
    };

    class GstDualSource {
    public:
        GstDualSource(std::string pipeline,
                      std::string id,
                      std::string sink_inf_name,
                      std::string sink_ui_name);

        bool start();
        void stop();

        bool read_frame(std::shared_ptr<FrameBundle>& out, int timeout_ms);

        const std::string& id() const { return id_; };

        ~GstDualSource();
    private:
        bool pull_frame_(GstElement* sink,
                           cv::Mat& out,
                           int64_t& pts_ns,
                           int timeout_ms);

        std::string pipeline_str_;
        std::string id_;
        std::string sink_inf_name_;
        std::string sink_ui_name_;

        GstElement* pipeline_ = nullptr;
        GstElement* sink_inf_ = nullptr;
        GstElement* sink_ui_ = nullptr;

        int64_t frame_id_ = 0;
    };
}