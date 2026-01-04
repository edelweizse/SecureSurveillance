#pragma once
#include <string>
#include <memory>
#include <vector>
#include <opencv2/core.hpp>

struct _GstElement;
using GstElement = _GstElement;

namespace ss {
    struct FramePacket {
        cv::Mat bgr;
        int64_t pts_ns = 0;
        int64_t frame_id = 0;
    };

    struct JpegPacket {
        std::shared_ptr<const std::vector<uint8_t>> jpeg;
        int64_t pts_ns = 0;
        int64_t frame_id = 0;
    };

    class GstDualSource {
    public:
        GstDualSource(std::string pipeline,
                      std::string id,
                      std::string sink_inf_name,
                      std::string sink_ui_name);

        bool start();
        void stop();

        bool read_inference(FramePacket& out, int timeout_ms);
        bool read_ui(JpegPacket& out, int timeout_ms);

        const std::string& id() const { return id_; };

        ~GstDualSource();
    private:
        bool pull_raw_bgr_(GstElement* sink, FramePacket& out, int timeout_ms);
        bool pull_jpeg_(GstElement* sink, JpegPacket& out, int timeout_ms);

        std::string pipeline_str_;
        std::string id_;
        std::string sink_inf_name_;
        std::string sink_ui_name_;

        GstElement* pipeline_ = nullptr;
        GstElement* sink_inf_ = nullptr;
        GstElement* sink_ui_ = nullptr;

        int64_t inf_frame_id_ = 0;
        int64_t ui_frame_id_ = 0;

        static bool gst_inited_;
    };
}