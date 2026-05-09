#include <pipeline/transforms.hpp>

#include <algorithm>
#include <array>

namespace veilsight {
    namespace {
        PointF apply(const cv::Matx23f& m, const PointF& p) {
            return PointF{
                m(0, 0) * p.x + m(0, 1) * p.y + m(0, 2),
                m(1, 0) * p.x + m(1, 1) * p.y + m(1, 2),
            };
        }
    }

    SpatialTransform identity_transform(cv::Size size) {
        SpatialTransform transform;
        transform.source_size = size;
        transform.target_size = size;
        transform.source_to_target = cv::Matx23f::eye();
        transform.target_to_source = cv::Matx23f::eye();
        return transform;
    }

    PointF map_point(const SpatialTransform& transform, const PointF& point) {
        return apply(transform.source_to_target, point);
    }

    RectF map_rect(const SpatialTransform& transform, const RectF& rect) {
        const std::array<PointF, 4> corners = {
            PointF{rect.x, rect.y},
            PointF{rect.x + rect.w, rect.y},
            PointF{rect.x, rect.y + rect.h},
            PointF{rect.x + rect.w, rect.y + rect.h},
        };

        auto mapped = map_point(transform, corners[0]);
        float min_x = mapped.x;
        float max_x = mapped.x;
        float min_y = mapped.y;
        float max_y = mapped.y;
        for (size_t i = 1; i < corners.size(); ++i) {
            mapped = map_point(transform, corners[i]);
            min_x = std::min(min_x, mapped.x);
            max_x = std::max(max_x, mapped.x);
            min_y = std::min(min_y, mapped.y);
            max_y = std::max(max_y, mapped.y);
        }

        return RectF{min_x, min_y, max_x - min_x, max_y - min_y};
    }

    FaceObservation map_face(const SpatialTransform& transform, const FaceObservation& face) {
        FaceObservation out = face;
        out.bbox = map_rect(transform, face.bbox);
        for (int i = 0; i < out.landmark_count && i < 5; ++i) {
            out.landmarks[static_cast<size_t>(i)] = map_point(transform, face.landmarks[static_cast<size_t>(i)]);
        }
        return out;
    }

    Box map_box(const SpatialTransform& transform, const Box& box) {
        Box out = box;
        const RectF mapped = map_rect(transform, RectF{box.x, box.y, box.w, box.h});
        out.x = mapped.x;
        out.y = mapped.y;
        out.w = mapped.w;
        out.h = mapped.h;
        if (out.face) {
            out.face = map_face(transform, *out.face);
        }
        return out;
    }
}
