#pragma once

#include <common/config.hpp>
#include <pipeline/types.hpp>

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

namespace veilsight {
    class IPersonDetector {
    public:
        virtual ~IPersonDetector() = default;
        virtual std::vector<Box> detect(const cv::Mat& bgr) = 0;
    };

    class IPersonDetectorFactory {
    public:
        virtual ~IPersonDetectorFactory() = default;
        virtual std::unique_ptr<IPersonDetector> create() const = 0;
        virtual int backend_threads() const = 0;
    };

    std::unique_ptr<IPersonDetectorFactory> create_person_detector_factory(const PersonDetectorModuleConfig& cfg);
    std::unique_ptr<IPersonDetector> create_person_detector(const PersonDetectorModuleConfig& cfg);
}
