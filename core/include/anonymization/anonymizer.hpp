#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <pipeline/types.hpp>

namespace ss {
    struct AnonymizerConfig {
        // Supported methods: "pixelate", "blur"
        std::string method = "pixelate";

        // Pixelate: downscale ROI by this factor, then upsample with nearest-neighbor.
        int pixelation_divisor = 10;

        // Blur: gaussian kernel size (will be forced to odd and >= 3).
        int blur_kernel = 31;
    };

    class Anonymizer {
    public:
        explicit Anonymizer(AnonymizerConfig cfg);

        void apply(cv::Mat& ui_frame,
                   const std::vector<Box>& boxes_inf_space,
                   float sx,
                   float sy,
                   float tx,
                   float ty) const;

    private:
        enum class Method {
            Pixelate,
            Blur
        };

        cv::Rect map_box_to_ui_(const Box& b,
                                float sx,
                                float sy,
                                float tx,
                                float ty,
                                int ui_w,
                                int ui_h) const;

        void apply_pixelate_(cv::Mat& roi) const;
        void apply_blur_(cv::Mat& roi) const;

        Method method_ = Method::Pixelate;
        int pixelation_divisor_ = 10;
        int blur_kernel_ = 31;
    };
}
