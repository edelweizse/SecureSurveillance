#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace veilsight {
    struct PointF {
        float x = 0.0f;
        float y = 0.0f;
    };

    struct RectF {
        float x = 0.0f;
        float y = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
    };

    enum class FaceSource {
        FullFrame,
        PersonRoi,
        Predicted
    };

    struct FaceObservation {
        RectF bbox;
        std::array<PointF, 5> landmarks;
        int landmark_count = 0;
        float score = 0.0f;
        int64_t frame_id = 0;
        FaceSource source = FaceSource::PersonRoi;
        bool fresh = false;
    };

    struct Box {
        float x = 0.0f;
        float y = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
        int id = -1;
        float score = 0.0f;
        bool occluded = false;
        std::string identity_key = "";
        float identity_confidence = 0.0f;
        std::string privacy_action = "anonymize";
        std::string recognition_state = "";
        std::optional<FaceObservation> face;
    };

    struct FrameCtx {
        std::string stream_id;
        int64_t frame_id = 0;
        int64_t pts_ns = 0;
        uint64_t created_steady_ns = 0;

        // Map boxes from inference frame coordinates into UI frame coordinates:
        // ui = inf * scale + offset
        float scale_x = 1.0f;
        float scale_y = 1.0f;
        float offset_x = 0.0f;
        float offset_y = 0.0f;

        int inf_w = 0;
        int inf_h = 0;
        int ui_w = 0;
        int ui_h = 0;

        cv::Mat ui;  // will be mutated by anonymizer and output to user
        cv::Mat inf; // will be released after inference
        std::vector<Box> tracked_boxes;
    };

    using FramePtr = std::shared_ptr<FrameCtx>;

}
