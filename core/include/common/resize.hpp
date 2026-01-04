#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>

namespace ss {
    inline int interp_from_str(const std::string& s) {
        if (s == "nearest") return cv::INTER_NEAREST;
        if (s == "cubic") return cv::INTER_CUBIC;
        if (s == "area") return cv::INTER_AREA;
        return cv::INTER_LINEAR;
    }

    inline cv::Mat resize_frame(
        const cv::Mat& src,
        int target_w,
        int target_h,
        bool keep_aspect,
        int interp
    ) {
        if (target_w <= 0 || target_h <= 0) return src;

        if (!keep_aspect) {
            cv::Mat dst;
            cv::resize(src, dst, {target_w, target_h}, 0, 0, interp);
            return dst;
        }

        float sx = float(target_w) / float(src.cols);
        float sy = float(target_h) / float(src.rows);
        float s = std::min(sx, sy);

        int new_w = std::max(1, int(src.cols * s));
        int new_h = std::max(1, int(src.rows * s));

        cv::Mat resized;
        cv::resize(src, resized, {new_w, new_h}, 0, 0, interp);

        cv::Mat out(target_h, target_w, src.type(), cv::Scalar::all(0));
        int x = (target_w - new_w) / 2;
        int y = (target_h - new_h) / 2;
        resized.copyTo(out((cv::Rect(x, y, new_w, new_h))));
        return out;
    }
}