#pragma once

#include <common/config.hpp>
#include <pipeline/types.hpp>

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace veilsight {
    struct FaceDetectorRunConfig {
        int input_w = 320;
        int input_h = 320;
    };

    class IFaceDetector {
    public:
        virtual ~IFaceDetector() = default;

        virtual std::vector<FaceObservation> detect_faces(
            const cv::Mat& bgr,
            const FaceDetectorRunConfig& run) = 0;
    };

    class IFaceDetectorFactory {
    public:
        virtual ~IFaceDetectorFactory() = default;

        virtual std::unique_ptr<IFaceDetector> create() const = 0;
        virtual int backend_threads() const = 0;
    };

    std::unique_ptr<IFaceDetectorFactory> create_face_detector_factory(
        const FaceDetectorModuleConfig& cfg);

    std::unique_ptr<IFaceDetector> create_face_detector(
        const FaceDetectorModuleConfig& cfg);
}
