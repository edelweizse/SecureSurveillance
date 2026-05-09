#pragma once

#include <common/config.hpp>
#include <face_detector/face_detector.hpp>
#include <pipeline/tasks.hpp>
#include <pipeline/types.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

namespace veilsight {
    struct FaceTrackCacheEntry {
        RectF face_rel;
        bool has_face_rel = false;
        FaceObservation last_face;
        int64_t last_seen_frame = -1;
        int64_t last_face_frame = -1;
        int64_t last_probe_frame = -1;
        int64_t next_probe_frame = 0;
        int miss_retry = 0;
    };

    struct FaceStreamCache {
        std::unordered_map<int, FaceTrackCacheEntry> tracks;
        int64_t last_full_frame_sweep = -1;
    };

    class FaceStateStore {
    public:
        std::mutex mutex;
        std::unordered_map<std::string, FaceStreamCache> streams;
    };

    struct FaceRoiCandidate {
        size_t track_index = 0;
        cv::Rect roi;
        float priority = 0.0f;
    };

    cv::Rect make_face_roi(const Box& person,
                           int frame_w,
                           int frame_h,
                           const FacePolicyConfig& cfg);

    FaceObservation map_face_from_roi(const FaceObservation& face,
                                      const cv::Rect& roi,
                                      int64_t frame_id,
                                      FaceSource source);

    std::optional<FaceObservation> predict_face_from_cache(const FaceTrackCacheEntry& entry,
                                                           const Box& person,
                                                           int64_t frame_id);

    class HybridFacePolicy {
    public:
        explicit HybridFacePolicy(FacePolicyConfig cfg,
                                  std::shared_ptr<FaceStateStore> state = std::make_shared<FaceStateStore>());

        void annotate(FrameCtx& frame,
                      std::vector<Box>& tracks,
                      IFaceDetector& detector);

        std::shared_ptr<FaceStateStore> state_store() const {
            return state_;
        }

    private:
        FacePolicyConfig cfg_;
        std::shared_ptr<FaceStateStore> state_;
    };

    class FaceProbePlanner {
    public:
        explicit FaceProbePlanner(FacePolicyConfig cfg,
                                  std::shared_ptr<FaceStateStore> state = std::make_shared<FaceStateStore>());

        std::vector<FaceDetectionTask> plan(FrameCtx& frame, std::vector<Box>& tracks);

        std::shared_ptr<FaceStateStore> state_store() const {
            return state_;
        }

    private:
        FacePolicyConfig cfg_;
        std::shared_ptr<FaceStateStore> state_;
    };

    class FaceResultApplier {
    public:
        explicit FaceResultApplier(FacePolicyConfig cfg,
                                   std::shared_ptr<FaceStateStore> state = std::make_shared<FaceStateStore>());

        void apply(FrameCtx& frame,
                   std::vector<Box>& tracks,
                   const std::vector<FaceDetectionResult>& results);

        std::shared_ptr<FaceStateStore> state_store() const {
            return state_;
        }

    private:
        FacePolicyConfig cfg_;
        std::shared_ptr<FaceStateStore> state_;
    };
}
