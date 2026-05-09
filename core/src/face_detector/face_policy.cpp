#include <face_detector/face_policy.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
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

        RectF upper_region(const Box& person, const FacePolicyConfig& cfg) {
            const float expand = std::max(0.0f, cfg.roi_width_expand_ratio);
            const float top_pad = std::max(0.0f, cfg.roi_top_pad_ratio);
            const float x = person.x - person.w * expand;
            const float y = person.y - person.h * top_pad;
            const float w = person.w * (1.0f + 2.0f * expand);
            const float h = person.h * (std::clamp(cfg.roi_height_ratio, 0.05f, 1.0f) + top_pad);
            return RectF{x, y, w, h};
        }

        bool center_inside(const RectF& inner, const RectF& outer) {
            const float cx = inner.x + inner.w * 0.5f;
            const float cy = inner.y + inner.h * 0.5f;
            return cx >= outer.x && cx <= outer.x + outer.w &&
                   cy >= outer.y && cy <= outer.y + outer.h;
        }

        bool plausible_face_for_track(const FaceObservation& face,
                                      const Box& person,
                                      const FacePolicyConfig& cfg) {
            if (face.score < cfg.min_face_score) return false;
            if (face.bbox.w < 2.0f || face.bbox.h < 2.0f || person.w <= 1.0f || person.h <= 1.0f) {
                return false;
            }

            const RectF upper = upper_region(person, cfg);
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
                               const Box& person,
                               const FacePolicyConfig& cfg) {
            const RectF upper = upper_region(person, cfg);
            const float face_area = std::max(1.0f, area_of(face.bbox));
            const float containment = intersection_area(face.bbox, upper) / face_area;
            return face.score + containment * 2.0f + iou_of(face.bbox, upper);
        }

        void store_face(FaceTrackCacheEntry& entry,
                        const Box& person,
                        const FaceObservation& face,
                        int64_t frame_id) {
            if (person.w <= 1.0f || person.h <= 1.0f) return;
            entry.face_rel.x = (face.bbox.x - person.x) / person.w;
            entry.face_rel.y = (face.bbox.y - person.y) / person.h;
            entry.face_rel.w = face.bbox.w / person.w;
            entry.face_rel.h = face.bbox.h / person.h;
            entry.has_face_rel = true;
            entry.last_face = face;
            entry.last_face_frame = frame_id;
            entry.miss_retry = 0;
            entry.next_probe_frame = frame_id + 1;
        }

        bool needs_refresh(const FaceTrackCacheEntry& entry,
                           const Box& track,
                           int64_t frame_id,
                           const FacePolicyConfig& cfg) {
            if (track.face && track.face->fresh && track.face->frame_id == frame_id) return false;
            if (!entry.has_face_rel || entry.last_face_frame < 0) return true;
            return frame_id - entry.last_face_frame >= std::max(1, cfg.refresh_interval);
        }

        void register_roi_miss(FaceTrackCacheEntry& entry,
                               int64_t frame_id,
                               const FacePolicyConfig& cfg) {
            const int initial = std::max(1, cfg.miss_retry_initial);
            const int maximum = std::max(initial, cfg.miss_retry_max);
            entry.miss_retry = entry.miss_retry > 0 ? std::min(maximum, entry.miss_retry * 2) : initial;
            entry.last_probe_frame = frame_id;
            entry.next_probe_frame = frame_id + entry.miss_retry;
        }

        void assign_full_frame_faces(std::vector<Box>& tracks,
                                     FaceStreamCache& stream,
                                     std::vector<FaceObservation> faces,
                                     int64_t frame_id,
                                     const FacePolicyConfig& cfg) {
            std::sort(faces.begin(), faces.end(), [](const FaceObservation& a, const FaceObservation& b) {
                return a.score > b.score;
            });

            std::vector<int> assigned_per_track(tracks.size(), 0);
            const int max_per_track = std::max(1, cfg.max_faces_per_track);

            for (auto face : faces) {
                face.frame_id = frame_id;
                face.source = FaceSource::FullFrame;
                face.fresh = true;

                float best_score = -std::numeric_limits<float>::infinity();
                size_t best_track = tracks.size();
                for (size_t i = 0; i < tracks.size(); ++i) {
                    if (assigned_per_track[i] >= max_per_track) continue;
                    if (!plausible_face_for_track(face, tracks[i], cfg)) continue;

                    const float score = face_track_score(face, tracks[i], cfg);
                    if (score > best_score) {
                        best_score = score;
                        best_track = i;
                    }
                }

                if (best_track >= tracks.size()) continue;
                tracks[best_track].face = face;
                ++assigned_per_track[best_track];

                const int id = tracks[best_track].id;
                if (id >= 0) {
                    store_face(stream.tracks[id], tracks[best_track], face, frame_id);
                }
            }
        }

        std::optional<FaceObservation> best_roi_face_for_track(const std::vector<FaceObservation>& faces,
                                                               const cv::Rect&,
                                                               const Box& track,
                                                               int64_t frame_id,
                                                               const FacePolicyConfig& cfg) {
            std::optional<FaceObservation> best;
            float best_score = -std::numeric_limits<float>::infinity();

            for (auto mapped : faces) {
                mapped.frame_id = frame_id;
                mapped.source = FaceSource::PersonRoi;
                mapped.fresh = true;
                if (!plausible_face_for_track(mapped, track, cfg)) continue;

                const float score = face_track_score(mapped, track, cfg);
                if (score > best_score) {
                    best_score = score;
                    best = mapped;
                }
            }

            return best;
        }

        int cleanup_ttl(const FacePolicyConfig& cfg) {
            return std::max({std::max(1, cfg.reuse_ttl),
                             std::max(1, cfg.miss_retry_max),
                             std::max(1, cfg.full_frame_interval)}) * 4;
        }
    } // namespace

    cv::Rect make_face_roi(const Box& person,
                           int frame_w,
                           int frame_h,
                           const FacePolicyConfig& cfg) {
        if (frame_w <= 0 || frame_h <= 0 || person.w <= 1.0f || person.h <= 1.0f) {
            return {};
        }

        const float expand = std::max(0.0f, cfg.roi_width_expand_ratio);
        const float top_pad = std::max(0.0f, cfg.roi_top_pad_ratio);
        const float roi_h_ratio = std::clamp(cfg.roi_height_ratio, 0.05f, 1.0f);

        const float x1 = person.x - person.w * expand;
        const float y1 = person.y - person.h * top_pad;
        const float x2 = person.x + person.w * (1.0f + expand);
        const float y2 = person.y + person.h * roi_h_ratio;

        const int ix1 = std::clamp(static_cast<int>(std::floor(x1)), 0, frame_w);
        const int iy1 = std::clamp(static_cast<int>(std::floor(y1)), 0, frame_h);
        const int ix2 = std::clamp(static_cast<int>(std::ceil(x2)), 0, frame_w);
        const int iy2 = std::clamp(static_cast<int>(std::ceil(y2)), 0, frame_h);

        if (ix2 <= ix1 || iy2 <= iy1) return {};
        return cv::Rect(ix1, iy1, ix2 - ix1, iy2 - iy1);
    }

    FaceObservation map_face_from_roi(const FaceObservation& face,
                                      const cv::Rect& roi,
                                      int64_t frame_id,
                                      FaceSource source) {
        SpatialTransform transform = identity_transform(roi.size());
        transform.target_size = roi.size();
        transform.source_to_target = cv::Matx23f(1.0f, 0.0f, static_cast<float>(roi.x),
                                                 0.0f, 1.0f, static_cast<float>(roi.y));
        transform.target_to_source = cv::Matx23f(1.0f, 0.0f, -static_cast<float>(roi.x),
                                                 0.0f, 1.0f, -static_cast<float>(roi.y));

        FaceObservation mapped = map_face(transform, face);
        mapped.frame_id = frame_id;
        mapped.source = source;
        mapped.fresh = source != FaceSource::Predicted;
        return mapped;
    }

    std::optional<FaceObservation> predict_face_from_cache(const FaceTrackCacheEntry& entry,
                                                           const Box& person,
                                                           int64_t frame_id) {
        if (!entry.has_face_rel || person.w <= 1.0f || person.h <= 1.0f) return std::nullopt;

        FaceObservation predicted = entry.last_face;
        predicted.bbox.x = person.x + entry.face_rel.x * person.w;
        predicted.bbox.y = person.y + entry.face_rel.y * person.h;
        predicted.bbox.w = entry.face_rel.w * person.w;
        predicted.bbox.h = entry.face_rel.h * person.h;
        predicted.landmark_count = 0;
        predicted.frame_id = frame_id;
        predicted.source = FaceSource::Predicted;
        predicted.fresh = false;
        return predicted;
    }

    namespace {
        void validate_policy_mode(FacePolicyConfig& cfg) {
            if (cfg.mode.empty()) cfg.mode = "hybrid";
            if (cfg.mode != "hybrid") {
                throw std::invalid_argument("[FacePolicy] Unsupported face policy mode: " + cfg.mode);
            }
        }

        std::string full_frame_probe_id(int64_t frame_id) {
            return std::to_string(frame_id) + ":full";
        }

        std::string roi_probe_id(int64_t frame_id, size_t track_index) {
            return std::to_string(frame_id) + ":roi:" + std::to_string(track_index);
        }
    }

    FaceProbePlanner::FaceProbePlanner(FacePolicyConfig cfg,
                                       std::shared_ptr<FaceStateStore> state)
        : cfg_(std::move(cfg)),
          state_(std::move(state)) {
        if (!state_) state_ = std::make_shared<FaceStateStore>();
        validate_policy_mode(cfg_);
    }

    std::vector<FaceDetectionTask> FaceProbePlanner::plan(FrameCtx& frame, std::vector<Box>& tracks) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        std::vector<FaceDetectionTask> tasks;

        FaceStreamCache& stream = state_->streams[frame.stream_id];
        std::unordered_set<int> current_ids;
        current_ids.reserve(tracks.size());
        if (tracks.empty()) {
            const int ttl = cleanup_ttl(cfg_);
            for (auto it = stream.tracks.begin(); it != stream.tracks.end();) {
                const bool stale = it->second.last_seen_frame >= 0 && frame.frame_id - it->second.last_seen_frame > ttl;
                if (stale) {
                    it = stream.tracks.erase(it);
                } else {
                    ++it;
                }
            }
            return tasks;
        }

        int missing_or_stale = 0;
        for (size_t i = 0; i < tracks.size(); ++i) {
            Box& track = tracks[i];
            track.privacy_action = "anonymize";
            track.face.reset();

            if (track.id >= 0) {
                current_ids.insert(track.id);
                FaceTrackCacheEntry& entry = stream.tracks[track.id];
                if (entry.has_face_rel &&
                    entry.last_face_frame >= 0 &&
                    frame.frame_id - entry.last_face_frame <= std::max(0, cfg_.reuse_ttl)) {
                    track.face = predict_face_from_cache(entry, track, frame.frame_id);
                }
                if (needs_refresh(entry, track, frame.frame_id, cfg_)) {
                    ++missing_or_stale;
                }
            } else {
                ++missing_or_stale;
            }
        }

        const int full_interval = std::max(1, cfg_.full_frame_interval);
        const int roi_budget = std::max(0, cfg_.max_roi_probes_per_frame);
        const bool interval_sweep = (frame.frame_id % full_interval) == 0;
        const int saturated_interval = std::max(1, full_interval / 2);
        const bool saturated_sweep =
            missing_or_stale > roi_budget &&
            (stream.last_full_frame_sweep < 0 ||
             frame.frame_id - stream.last_full_frame_sweep >= saturated_interval);

        if (!frame.inf.empty() && (interval_sweep || saturated_sweep)) {
            FaceDetectionTask task;
            task.stream_id = frame.stream_id;
            task.frame_id = frame.frame_id;
            task.probe_id = full_frame_probe_id(frame.frame_id);
            task.kind = FaceProbeKind::FullFrame;
            task.input.image = frame.inf;
            task.input.image_to_frame = identity_transform(frame.inf.size());
            task.run.input_w = std::max(1, cfg_.full_frame_input_w);
            task.run.input_h = std::max(1, cfg_.full_frame_input_h);
            tasks.push_back(std::move(task));
            stream.last_full_frame_sweep = frame.frame_id;
        }

        std::vector<FaceRoiCandidate> candidates;
        candidates.reserve(tracks.size());
        const float frame_cx = static_cast<float>(std::max(1, frame.inf_w)) * 0.5f;
        const float frame_cy = static_cast<float>(std::max(1, frame.inf_h)) * 0.5f;
        const float frame_diag = std::max(1.0f, std::hypot(frame_cx, frame_cy));

        for (size_t i = 0; i < tracks.size(); ++i) {
            Box& track = tracks[i];
            if (track.h < static_cast<float>(std::max(1, cfg_.min_track_height))) continue;

            FaceTrackCacheEntry* entry = nullptr;
            bool is_new = false;
            if (track.id >= 0) {
                entry = &stream.tracks[track.id];
                is_new = entry->last_seen_frame < 0 || frame.frame_id - entry->last_seen_frame > 1;
                if (!needs_refresh(*entry, track, frame.frame_id, cfg_)) {
                    entry->last_seen_frame = frame.frame_id;
                    continue;
                }
                if (frame.frame_id < entry->next_probe_frame) {
                    entry->last_seen_frame = frame.frame_id;
                    continue;
                }
            } else if (track.face && track.face->fresh) {
                continue;
            }

            const cv::Rect roi = make_face_roi(track, frame.inf_w, frame.inf_h, cfg_);
            if (roi.width < 2 || roi.height < 2) continue;

            const float tcx = track.x + track.w * 0.5f;
            const float tcy = track.y + track.h * 0.5f;
            const float center_score = 1.0f - std::min(1.0f, std::hypot(tcx - frame_cx, tcy - frame_cy) / frame_diag);
            const float size_score = std::min(1.0f, track.h / static_cast<float>(std::max(1, frame.inf_h)));
            const bool stale = entry ? (entry->last_face_frame < 0 ||
                                        frame.frame_id - entry->last_face_frame >= std::max(1, cfg_.refresh_interval))
                                     : true;

            FaceRoiCandidate c;
            c.track_index = i;
            c.roi = roi;
            c.priority = (is_new ? 100000.0f : 0.0f) +
                         (stale ? 50000.0f : 0.0f) +
                         size_score * 1000.0f +
                         center_score * 100.0f;
            candidates.push_back(c);
        }

        std::sort(candidates.begin(), candidates.end(), [](const FaceRoiCandidate& a, const FaceRoiCandidate& b) {
            return a.priority > b.priority;
        });

        int probes = 0;
        for (const FaceRoiCandidate& candidate : candidates) {
            if (probes >= roi_budget || frame.inf.empty()) break;
            Box& track = tracks[candidate.track_index];

            FaceDetectionTask task;
            task.stream_id = frame.stream_id;
            task.frame_id = frame.frame_id;
            task.probe_id = roi_probe_id(frame.frame_id, candidate.track_index);
            task.kind = FaceProbeKind::PersonRoi;
            task.track_index = candidate.track_index;
            task.roi = candidate.roi;
            task.input.image = frame.inf(candidate.roi).clone();
            task.input.image_to_frame = identity_transform(candidate.roi.size());
            task.input.image_to_frame.target_size = frame.inf.size();
            task.input.image_to_frame.source_to_target =
                cv::Matx23f(1.0f, 0.0f, static_cast<float>(candidate.roi.x),
                            0.0f, 1.0f, static_cast<float>(candidate.roi.y));
            task.input.image_to_frame.target_to_source =
                cv::Matx23f(1.0f, 0.0f, -static_cast<float>(candidate.roi.x),
                            0.0f, 1.0f, -static_cast<float>(candidate.roi.y));
            task.run.input_w = std::max(1, cfg_.roi_input_w);
            task.run.input_h = std::max(1, cfg_.roi_input_h);
            tasks.push_back(std::move(task));
            ++probes;

            FaceTrackCacheEntry* entry = track.id >= 0 ? &stream.tracks[track.id] : nullptr;
            if (entry) {
                entry->last_probe_frame = frame.frame_id;
                entry->next_probe_frame = frame.frame_id + 1;
            }
        }

        for (auto& track : tracks) {
            if (track.id >= 0) {
                stream.tracks[track.id].last_seen_frame = frame.frame_id;
            }
        }

        const int ttl = cleanup_ttl(cfg_);
        for (auto it = stream.tracks.begin(); it != stream.tracks.end();) {
            const bool current = current_ids.find(it->first) != current_ids.end();
            const bool stale = it->second.last_seen_frame >= 0 && frame.frame_id - it->second.last_seen_frame > ttl;
            if (!current && stale) {
                it = stream.tracks.erase(it);
            } else {
                ++it;
            }
        }

        return tasks;
    }

    FaceResultApplier::FaceResultApplier(FacePolicyConfig cfg,
                                         std::shared_ptr<FaceStateStore> state)
        : cfg_(std::move(cfg)),
          state_(std::move(state)) {
        if (!state_) state_ = std::make_shared<FaceStateStore>();
        validate_policy_mode(cfg_);
    }

    void FaceResultApplier::apply(FrameCtx& frame,
                                  std::vector<Box>& tracks,
                                  const std::vector<FaceDetectionResult>& results) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        FaceStreamCache& stream = state_->streams[frame.stream_id];

        for (const auto& result : results) {
            if (result.kind != FaceProbeKind::FullFrame) continue;
            assign_full_frame_faces(tracks, stream, result.faces, frame.frame_id, cfg_);
        }

        for (const auto& result : results) {
            if (result.kind != FaceProbeKind::PersonRoi) continue;
            if (result.track_index >= tracks.size()) continue;
            Box& track = tracks[result.track_index];
            auto best = best_roi_face_for_track(result.faces, result.roi, track, frame.frame_id, cfg_);
            FaceTrackCacheEntry* entry = track.id >= 0 ? &stream.tracks[track.id] : nullptr;
            if (best) {
                track.face = *best;
                if (entry) {
                    store_face(*entry, track, *best, frame.frame_id);
                }
            } else if (entry) {
                register_roi_miss(*entry, frame.frame_id, cfg_);
            }
        }
    }

    HybridFacePolicy::HybridFacePolicy(FacePolicyConfig cfg,
                                       std::shared_ptr<FaceStateStore> state)
        : cfg_(std::move(cfg)),
          state_(std::move(state)) {
        if (!state_) state_ = std::make_shared<FaceStateStore>();
        validate_policy_mode(cfg_);
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
            result.track_index = task.track_index;
            result.roi = task.roi;
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
