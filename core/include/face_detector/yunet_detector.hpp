#pragma once

#include <person_detector/person_detector.hpp>
#include <face_detector/face_detector.hpp>

#include <memory>

namespace veilsight {
    class YuNetDetector final : public IPersonDetector, public IFaceDetector {
    public:
        explicit YuNetDetector(YuNetModuleConfig cfg);
        ~YuNetDetector();

        YuNetDetector(YuNetDetector&&) noexcept;
        YuNetDetector& operator=(YuNetDetector&&) noexcept;

        YuNetDetector(const YuNetDetector&) = delete;
        YuNetDetector& operator=(const YuNetDetector&) = delete;

        std::vector<Box> detect(const cv::Mat& bgr) override;
        std::vector<FaceObservation> detect_faces(const cv::Mat& bgr,
                                                  const FaceDetectorRunConfig& run) override;

    private:
        YuNetModuleConfig cfg_;
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}
