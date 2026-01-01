#pragma once

#include <string>
#include <opencv2/core.hpp>

namespace ss {
    struct FramePacket {
        cv::Mat bgr;
        int64_t pts_ns = 0;
        int64_t frame_id = 0;
        std::string src_id;
        int width = 0;
        int height = 0;
    };
}