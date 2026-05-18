#pragma once

#include <person_detector/person_detector.hpp>

#include <memory>

namespace veilsight {
    class UhdDetector final : public IPersonDetector {
    public:
        explicit UhdDetector(UhdModuleConfig cfg);
        ~UhdDetector();

        UhdDetector(UhdDetector&&) noexcept;
        UhdDetector& operator=(UhdDetector&&) noexcept;

        UhdDetector(const UhdDetector&) = delete;
        UhdDetector& operator=(const UhdDetector&) = delete;

        std::vector<Box> detect(const cv::Mat& bgr) override;

    private:
        UhdModuleConfig cfg_;
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}
