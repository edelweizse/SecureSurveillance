#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ss {
    // placeholder
    struct Box {
        int x = 0, y = 0, w = 0, h = 0;
        int id = -1;
    };
    struct FrameCtx {
        std::string stream_id;
        int64_t frame_id = 0;
        int64_t pts_ns = 0;

        float scale_x = 0.0f;
        float scale_y = 0.0f;

        cv::Mat ui;  // will be mutated by anonymizer and output to user
        cv::Mat inf; // will be released after inference
    };

    using FramePtr = std::shared_ptr<FrameCtx>;

    // placeholder
    struct InferResults {
        std::string stream_id;
        int64_t frame_id = 0;
        std::vector<Box> bboxes;
    };
}