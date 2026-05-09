#pragma once

#include <person_detector/person_detector.hpp>

#include <memory>

namespace veilsight {
    class YoloXDetector final : public IPersonDetector {
    public:
        explicit YoloXDetector(YoloXModuleConfig cfg);
        ~YoloXDetector();

        YoloXDetector(YoloXDetector&&) noexcept;
        YoloXDetector& operator=(YoloXDetector&&) noexcept;

        YoloXDetector(const YoloXDetector&) = delete;
        YoloXDetector& operator=(const YoloXDetector&) = delete;

        std::vector<Box> detect(const cv::Mat& bgr) override;

    private:
        YoloXModuleConfig cfg_;
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}
