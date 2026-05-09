#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <face_detector/face_detector.hpp>
#include <pipeline/transforms.hpp>
#include <pipeline/types.hpp>

namespace veilsight {
    struct PersonDetectionTask {
        std::string stream_id;
        int64_t frame_id = 0;
        StageImage input;
        FramePtr frame_ctx;
    };

    struct PersonDetectionResult {
        std::string stream_id;
        int64_t frame_id = 0;
        std::vector<Box> boxes;
    };

    enum class FaceProbeKind {
        FullFrame
    };

    struct FaceDetectionTask {
        std::string stream_id;
        int64_t frame_id = 0;
        std::string probe_id;
        FaceProbeKind kind = FaceProbeKind::FullFrame;
        StageImage input;
        FaceDetectorRunConfig run;
    };

    struct FaceDetectionResult {
        std::string stream_id;
        int64_t frame_id = 0;
        std::string probe_id;
        FaceProbeKind kind = FaceProbeKind::FullFrame;
        std::vector<FaceObservation> faces;
    };

    struct RecognitionTask {
        std::string stream_id;
        int64_t frame_id = 0;
        FramePtr frame;
        std::vector<Box> tracks;
    };

    struct RecognitionResult {
        std::string stream_id;
        int64_t frame_id = 0;
        FramePtr frame;
        std::vector<Box> tracks;
    };

    struct IdentityTask {
        std::string stream_id;
        int64_t frame_id = 0;
        FramePtr frame;
        std::vector<Box> tracks;
    };

    struct IdentityResult {
        std::string stream_id;
        int64_t frame_id = 0;
        FramePtr frame;
        std::vector<Box> tracks;
    };

    struct AnonymizeTask {
        std::string stream_id;
        int64_t frame_id = 0;
        FramePtr frame;
    };

    struct AnonymizeResult {
        std::string stream_id;
        int64_t frame_id = 0;
        FramePtr frame;
    };
}
