#pragma once

#include <person_detector/person_detector.hpp>
#include <face_detector/face_detector.hpp>

#include <memory>

namespace veilsight {
    class SCRFDDetector final : public IPersonDetector, public IFaceDetector {
    public:
        explicit SCRFDDetector(SCRFDModuleConfig cfg);
        ~SCRFDDetector();

        SCRFDDetector(SCRFDDetector&&) noexcept;
        SCRFDDetector& operator=(SCRFDDetector&&) noexcept;

        SCRFDDetector(const SCRFDDetector&) = delete;
        SCRFDDetector& operator=(const SCRFDDetector&) = delete;

        std::vector<Box> detect(const cv::Mat& bgr) override;
        std::vector<FaceObservation> detect_faces(const cv::Mat& bgr,
                                                  const FaceDetectorRunConfig& run) override;

    private:
        SCRFDModuleConfig cfg_;
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}
