#include <anonymization/anonymizer.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace ss {
    namespace {
        std::string normalize_method(std::string s) {
            std::transform(s.begin(),
                           s.end(),
                           s.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }
    } // namespace

    Anonymizer::Anonymizer(AnonymizerConfig cfg) {
        const std::string method = normalize_method(std::move(cfg.method));
        if (method == "blur") {
            method_ = Method::Blur;
        } else {
            method_ = Method::Pixelate;
        }

        pixelation_divisor_ = std::max(2, cfg.pixelation_divisor);
        blur_kernel_ = std::max(3, cfg.blur_kernel);
        if ((blur_kernel_ % 2) == 0) blur_kernel_ += 1;
    }

    void Anonymizer::apply(cv::Mat& ui_frame,
                           const std::vector<Box>& boxes_inf_space,
                           float sx,
                           float sy,
                           float tx,
                           float ty) const {
        if (ui_frame.empty()) return;

        for (const auto& b : boxes_inf_space) {
            if (b.w <= 1.0f || b.h <= 1.0f) continue;

            const cv::Rect roi_rect =
                map_box_to_ui_(b, sx, sy, tx, ty, ui_frame.cols, ui_frame.rows);
            if (roi_rect.width < 2 || roi_rect.height < 2) continue;

            cv::Mat roi = ui_frame(roi_rect);
            if (method_ == Method::Blur) {
                apply_blur_(roi);
            } else {
                apply_pixelate_(roi);
            }
        }
    }

    cv::Rect Anonymizer::map_box_to_ui_(const Box& b,
                                        float sx,
                                        float sy,
                                        float tx,
                                        float ty,
                                        int ui_w,
                                        int ui_h) const {
        const int x = static_cast<int>(std::lround(b.x * sx + tx));
        const int y = static_cast<int>(std::lround(b.y * sy + ty));
        const int w = static_cast<int>(std::lround(b.w * sx));
        const int h = static_cast<int>(std::lround(b.h * sy));

        cv::Rect r(x, y, w, h);
        cv::Rect bounds(0, 0, ui_w, ui_h);
        r &= bounds;
        return r;
    }

    void Anonymizer::apply_pixelate_(cv::Mat& roi) const {
        cv::Mat tiny;
        const int tw = std::max(2, roi.cols / pixelation_divisor_);
        const int th = std::max(2, roi.rows / pixelation_divisor_);
        cv::resize(roi, tiny, cv::Size(tw, th), 0, 0, cv::INTER_LINEAR);
        cv::resize(tiny, roi, roi.size(), 0, 0, cv::INTER_NEAREST);
    }

    void Anonymizer::apply_blur_(cv::Mat& roi) const {
        cv::GaussianBlur(roi, roi, cv::Size(blur_kernel_, blur_kernel_), 0.0, 0.0);
    }
}
