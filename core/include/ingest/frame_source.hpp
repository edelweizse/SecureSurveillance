#pragma once

#include <opencv2/core.hpp>

namespace ss {

    struct FramePacket {
        cv::Mat bgr;
        int64_t pts_ns = 0;
        int64_t frame_id = 0;
    };

    struct IFrameSource {
        virtual ~IFrameSource() = default;
        virtual bool start() = 0;
        virtual void stop() = 0;
        virtual bool read(FramePacket& out, int timeout_ms) = 0;
        virtual const std::string& id() const = 0;
    };
}