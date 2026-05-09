#pragma once

#include <opencv2/core.hpp>

#include <pipeline/types.hpp>

namespace veilsight {
    struct SpatialTransform {
        cv::Size source_size;
        cv::Size target_size;
        cv::Matx23f source_to_target = cv::Matx23f::eye();
        cv::Matx23f target_to_source = cv::Matx23f::eye();
    };

    struct StageImage {
        cv::Mat image;
        SpatialTransform image_to_frame;
    };

    SpatialTransform identity_transform(cv::Size size);
    RectF map_rect(const SpatialTransform& transform, const RectF& rect);
    PointF map_point(const SpatialTransform& transform, const PointF& point);
    FaceObservation map_face(const SpatialTransform& transform, const FaceObservation& face);
    Box map_box(const SpatialTransform& transform, const Box& box);
}
