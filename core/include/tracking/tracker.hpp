#pragma once

#include <memory>
#include <vector>

#include <pipeline/types.hpp>

namespace ss {
    struct TrackerConfig {
        float high_thresh = 0.6f;
        float low_thresh = 0.2f;
        float match_iou_thresh = 0.3f;
        float low_match_iou_thresh = 0.2f;
        int min_hits = 2;
        int max_missed = 20;
    };

    class ITracker {
    public:
        virtual ~ITracker() = default;
        virtual std::vector<Box> update(const std::vector<Box>& detections) = 0;
    };

    std::unique_ptr<ITracker> create_demo_tracker(const TrackerConfig& cfg = {});
}
