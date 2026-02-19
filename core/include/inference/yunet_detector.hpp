#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <pipeline/types.hpp>

namespace ss {
    struct YuNetDetectorConfig {
        std::string param_path = "models/detector/face_detection_yunet_2023mar.ncnn.param";
        std::string bin_path = "models/detector/face_detection_yunet_2023mar.ncnn.bin";
        int input_w = 640;
        int input_h = 640;
        float score_threshold = 0.6f;
        float nms_threshold = 0.3f;
        int top_k = 750;
        int ncnn_threads = 1;
    };

    class YuNetDetector {
    public:
        explicit YuNetDetector(YuNetDetectorConfig cfg);
        ~YuNetDetector();

        YuNetDetector(YuNetDetector&&) noexcept;
        YuNetDetector& operator=(YuNetDetector&&) noexcept;

        YuNetDetector(const YuNetDetector&) = delete;
        YuNetDetector& operator=(const YuNetDetector&) = delete;

        std::vector<Box> detect(const cv::Mat& bgr) const;

    private:
        YuNetDetectorConfig cfg_;
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}
