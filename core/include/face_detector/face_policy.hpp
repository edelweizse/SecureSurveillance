#pragma once

#include <common/config.hpp>
#include <face_detector/face_detector.hpp>
#include <pipeline/tasks.hpp>
#include <pipeline/types.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace veilsight {
    class FaceStateStore {
    public:
        std::mutex mutex;
    };

    class HybridFacePolicy {
    public:
        explicit HybridFacePolicy(FaceDetectorModuleConfig cfg,
                                  std::shared_ptr<FaceStateStore> state = std::make_shared<FaceStateStore>());

        void annotate(FrameCtx& frame,
                      std::vector<Box>& tracks,
                      IFaceDetector& detector);

        std::shared_ptr<FaceStateStore> state_store() const {
            return state_;
        }

    private:
        FaceDetectorModuleConfig cfg_;
        std::shared_ptr<FaceStateStore> state_;
    };

    class FaceProbePlanner {
    public:
        explicit FaceProbePlanner(FaceDetectorModuleConfig cfg,
                                  std::shared_ptr<FaceStateStore> state = std::make_shared<FaceStateStore>());

        std::vector<FaceDetectionTask> plan(FrameCtx& frame, std::vector<Box>& tracks);

        std::shared_ptr<FaceStateStore> state_store() const {
            return state_;
        }

    private:
        FaceDetectorModuleConfig cfg_;
        std::shared_ptr<FaceStateStore> state_;
    };

    class FaceResultApplier {
    public:
        explicit FaceResultApplier(FaceDetectorModuleConfig cfg,
                                   std::shared_ptr<FaceStateStore> state = std::make_shared<FaceStateStore>());

        void apply(FrameCtx& frame,
                   std::vector<Box>& tracks,
                   const std::vector<FaceDetectionResult>& results);

        std::shared_ptr<FaceStateStore> state_store() const {
            return state_;
        }

    private:
        FaceDetectorModuleConfig cfg_;
        std::shared_ptr<FaceStateStore> state_;
    };
}
