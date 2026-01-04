#pragma once

#include <ingest/frame_source.hpp>
#include <string>

struct _GstElement;
using GstElement = _GstElement;

namespace ss {
    class GstFrameSource: public IFrameSource {
    public:
        GstFrameSource(std::string pipeline, std::string src_id, std::string sink_name);

        bool start() override;
        void stop() override;
        bool read(FramePacket& out, int timeout_ms = 1000) override;
        const std::string& id() const override { return id_ ;};

        ~GstFrameSource() override;

    private:
        std::string pipeline_str_;
        std::string id_;
        std::string sink_name_;

        GstElement* pipeline_ = nullptr;
        GstElement* sink_ = nullptr;

        int64_t frame_id_ = 0;

        static bool gst_inited_;
    };
}