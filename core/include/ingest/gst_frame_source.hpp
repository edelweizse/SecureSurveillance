#pragma once

#include <ingest/frame_source.hpp>
#include <string>

typedef struct _GstElement GstElement;

namespace ss {
    class GstFrameSource: public IFrameSource {
    public:
        GstFrameSource(std::string pipeline, std::string src_id = "source");

        bool start() override;
        bool read(FramePacket& out, int timeout_ms = 1000) override;
        void stop() override;

        ~GstFrameSource() override;
    private:
        std::string pipeline_str_;
        std::string src_id_;

        GstElement* pipeline_ = nullptr;
        GstElement* sink_ = nullptr;

        int64_t frame_id_ = 0;

        static bool gst_inited_;
    };
}