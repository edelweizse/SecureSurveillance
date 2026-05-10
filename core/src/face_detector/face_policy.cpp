#include <face_detector/face_policy.hpp>

#include <algorithm>
#include <limits>
#include <utility>

namespace veilsight {
    namespace {
        float area_of(const RectF& rect) {
            return std::max(0.0f, rect.w) * std::max(0.0f, rect.h);
        }

        float intersection_area(const RectF& a, const RectF& b) {
            const float x1 = std::max(a.x, b.x);
            const float y1 = std::max(a.y, b.y);
            const float x2 = std::min(a.x + a.w, b.x + b.w);
            const float y2 = std::min(a.y + a.h, b.y + b.h);
            return std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        }

        float iou_of(const RectF& a, const RectF& b) {
            const float inter = intersection_area(a, b);
            if (inter <= 0.0f) return 0.0f;
            const float uni = area_of(a) + area_of(b) - inter;
            return uni > 0.0f ? inter / uni : 0.0f;
        }

        RectF upper_region(const Box& person) {
            return RectF{
                person.x,
                person.y,
                person.w,
                person.h * 0.60f,
            };
        }

        bool center_inside(const RectF& inner, const RectF& outer) {
            const float cx = inner.x + inner.w * 0.5f;
            const float cy = inner.y + inner.h * 0.5f;
            return cx >= outer.x && cx <= outer.x + outer.w &&
                   cy >= outer.y && cy <= outer.y + outer.h;
        }

        bool plausible_face_for_track(const FaceObservation& face,
                                      const Box& person,
                                      float min_face_score) {
            if (face.score < min_face_score) return false;
            if (face.bbox.w < 2.0f || face.bbox.h < 2.0f || person.w <= 1.0f || person.h <= 1.0f) {
                return false;
            }

            const RectF upper = upper_region(person);
            const float face_area = area_of(face.bbox);
            if (face_area <= 0.0f) return false;

            const float containment = intersection_area(face.bbox, upper) / face_area;
            if (!center_inside(face.bbox, upper) || containment < 0.50f) {
                return false;
            }

            const float min_h = std::max(4.0f, person.h * 0.04f);
            const float max_h = std::max(min_h, person.h * 0.60f);
            if (face.bbox.h < min_h || face.bbox.h > max_h) return false;
            if (face.bbox.w > person.w * 0.95f) return false;
            return true;
        }

        float face_track_score(const FaceObservation& face,
                               const Box& person) {
            const RectF upper = upper_region(person);
            const float face_area = std::max(1.0f, area_of(face.bbox));
            const float containment = intersection_area(face.bbox, upper) / face_area;
            return face.score + containment * 2.0f + iou_of(face.bbox, upper);
        }

        Box face_only_box(const FaceObservation& face, int id) {
            Box box;
            box.x = face.bbox.x;
            box.y = face.bbox.y;
            box.w = face.bbox.w;
            box.h = face.bbox.h;
            box.id = id;
            box.score = face.score;
            box.privacy_action = "anonymize";
            box.recognition_state = "face_only";
            box.face = face;
            return box;
        }

        void assign_full_frame_faces(std::vector<Box>& tracks,
                                     std::vector<FaceObservation> faces,
                                     int64_t frame_id,
                                     bool emit_unassigned_faces,
                                     const FaceDetectorModuleConfig& cfg) {
            std::sort(faces.begin(), faces.end(), [](const FaceObservation& a, const FaceObservation& b) {
                return a.score > b.score;
            });

            std::vector<int> assigned_per_track(tracks.size(), 0);
            const size_t person_track_count = tracks.size();
            int face_only_id = -1;
            const int max_per_track = 1;
            const float min_face_score = cfg.type == "yunet" ? cfg.yunet.score_threshold : cfg.scrfd.score_threshold;

            for (auto face : faces) {
                face.frame_id = frame_id;
                face.source = FaceSource::FullFrame;
                face.fresh = true;

                float best_score = -std::numeric_limits<float>::infinity();
                size_t best_track = person_track_count;
                for (size_t i = 0; i < person_track_count; ++i) {
                    if (assigned_per_track[i] >= max_per_track) continue;
                    if (!plausible_face_for_track(face, tracks[i], min_face_score)) continue;

                    const float score = face_track_score(face, tracks[i]);
                    if (score > best_score) {
                        best_score = score;
                        best_track = i;
                    }
                }

                if (best_track >= person_track_count) {
                    if (!emit_unassigned_faces) continue;
                    tracks.push_back(face_only_box(face, face_only_id--));
                    continue;
                }
                tracks[best_track].face = face;
                ++assigned_per_track[best_track];
            }
        }

        FaceDetectorRunConfig run_config_for_detector(const FaceDetectorModuleConfig& cfg) {
            if (cfg.type == "yunet") {
                return FaceDetectorRunConfig{std::max(1, cfg.yunet.input_w), std::max(1, cfg.yunet.input_h)};
            }
            return FaceDetectorRunConfig{std::max(1, cfg.scrfd.input_w), std::max(1, cfg.scrfd.input_h)};
        }

        std::string full_frame_probe_id(int64_t frame_id) {
            return std::to_string(frame_id) + ":full";
        }
    } // namespace

    FaceProbePlanner::FaceProbePlanner(FaceDetectorModuleConfig cfg,
                                       std::shared_ptr<FaceStateStore> state)
        : cfg_(std::move(cfg)),
          state_(std::move(state)) {
        if (!state_) state_ = std::make_shared<FaceStateStore>();
    }

    std::vector<FaceDetectionTask> FaceProbePlanner::plan(FrameCtx& frame, std::vector<Box>& tracks) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        std::vector<FaceDetectionTask> tasks;

        for (auto& track : tracks) {
            track.privacy_action = "anonymize";
            track.face.reset();
        }

        if (frame.inf.empty()) return tasks;

        FaceDetectionTask task;
        task.stream_id = frame.stream_id;
        task.frame_id = frame.frame_id;
        task.probe_id = full_frame_probe_id(frame.frame_id);
        task.kind = FaceProbeKind::FullFrame;
        task.input.image = frame.inf;
        task.input.image_to_frame = identity_transform(frame.inf.size());
        task.run = run_config_for_detector(cfg_);
        tasks.push_back(std::move(task));

        return tasks;
    }

    FaceResultApplier::FaceResultApplier(FaceDetectorModuleConfig cfg,
                                         std::shared_ptr<FaceStateStore> state)
        : cfg_(std::move(cfg)),
          state_(std::move(state)) {
        if (!state_) state_ = std::make_shared<FaceStateStore>();
    }

    void FaceResultApplier::apply(FrameCtx& frame,
                                  std::vector<Box>& tracks,
                                  const std::vector<FaceDetectionResult>& results) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        for (const auto& result : results) {
            if (result.kind != FaceProbeKind::FullFrame) continue;
            assign_full_frame_faces(tracks, result.faces, frame.frame_id, frame.source_type == "webcam", cfg_);
        }
    }

    HybridFacePolicy::HybridFacePolicy(FaceDetectorModuleConfig cfg,
                                       std::shared_ptr<FaceStateStore> state)
        : cfg_(std::move(cfg)),
          state_(std::move(state)) {
        if (!state_) state_ = std::make_shared<FaceStateStore>();
    }

    void HybridFacePolicy::annotate(FrameCtx& frame,
                                    std::vector<Box>& tracks,
                                    IFaceDetector& detector) {
        FaceProbePlanner planner(cfg_, state_);
        FaceResultApplier applier(cfg_, state_);
        std::vector<FaceDetectionResult> results;
        for (const auto& task : planner.plan(frame, tracks)) {
            FaceDetectionResult result;
            result.stream_id = task.stream_id;
            result.frame_id = task.frame_id;
            result.probe_id = task.probe_id;
            result.kind = task.kind;
            const auto faces = detector.detect_faces(task.input.image, task.run);
            result.faces.reserve(faces.size());
            for (const auto& face : faces) {
                result.faces.push_back(map_face(task.input.image_to_frame, face));
            }
            results.push_back(std::move(result));
        }
        applier.apply(frame, tracks, results);
    }
}
