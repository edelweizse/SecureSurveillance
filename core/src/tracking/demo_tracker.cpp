#include <tracking/tracker.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace ss {
    namespace {
        struct TrackState {
            int id = -1;
            Box box{};
            int age = 0;
            int hits = 0;
            int missed = 0;

            float vx = 0.0f;
            float vy = 0.0f;
            float vw = 0.0f;
            float vh = 0.0f;
        };

        float area_of(const Box& b) {
            return std::max(0.0f, b.w) * std::max(0.0f, b.h);
        }

        float iou_of(const Box& a, const Box& b) {
            const float ax2 = a.x + a.w;
            const float ay2 = a.y + a.h;
            const float bx2 = b.x + b.w;
            const float by2 = b.y + b.h;

            const float xx1 = std::max(a.x, b.x);
            const float yy1 = std::max(a.y, b.y);
            const float xx2 = std::min(ax2, bx2);
            const float yy2 = std::min(ay2, by2);

            const float iw = std::max(0.0f, xx2 - xx1);
            const float ih = std::max(0.0f, yy2 - yy1);
            const float inter = iw * ih;
            if (inter <= 0.0f) return 0.0f;

            const float uni = area_of(a) + area_of(b) - inter;
            if (uni <= 0.0f) return 0.0f;
            return inter / uni;
        }

        class DemoTracker final : public ITracker {
        public:
            explicit DemoTracker(TrackerConfig cfg)
                : cfg_(std::move(cfg)) {}

            std::vector<Box> update(const std::vector<Box>& detections) override {
                for (auto& t : tracks_) {
                    t.age += 1;
                    t.missed += 1;
                    // Constant-velocity prediction keeps boxes alive across short occlusions.
                    t.box.x += t.vx;
                    t.box.y += t.vy;
                    t.box.w = std::max(1.0f, t.box.w + t.vw);
                    t.box.h = std::max(1.0f, t.box.h + t.vh);
                }

                std::vector<int> track_indices(tracks_.size());
                for (size_t i = 0; i < tracks_.size(); ++i) track_indices[i] = static_cast<int>(i);

                std::vector<int> high_det_indices;
                std::vector<int> low_det_indices;
                high_det_indices.reserve(detections.size());
                low_det_indices.reserve(detections.size());

                for (size_t i = 0; i < detections.size(); ++i) {
                    if (detections[i].score >= cfg_.high_thresh) {
                        high_det_indices.push_back(static_cast<int>(i));
                    } else if (detections[i].score >= cfg_.low_thresh) {
                        low_det_indices.push_back(static_cast<int>(i));
                    }
                }

                std::vector<int> unmatched_tracks;
                std::vector<int> unmatched_high_dets;
                match_greedy_(track_indices,
                              high_det_indices,
                              detections,
                              cfg_.match_iou_thresh,
                              unmatched_tracks,
                              unmatched_high_dets);

                std::vector<int> unmatched_tracks_after_low;
                std::vector<int> unused_low_dets;
                match_greedy_(unmatched_tracks,
                              low_det_indices,
                              detections,
                              cfg_.low_match_iou_thresh,
                              unmatched_tracks_after_low,
                              unused_low_dets);

                for (int di : unmatched_high_dets) {
                    TrackState t;
                    t.id = next_track_id_++;
                    t.box = detections[static_cast<size_t>(di)];
                    t.box.id = t.id;
                    t.box.occluded = false;
                    t.age = 1;
                    t.hits = 1;
                    t.missed = 0;
                    tracks_.push_back(std::move(t));
                }

                tracks_.erase(
                    std::remove_if(tracks_.begin(),
                                   tracks_.end(),
                                   [this](const TrackState& t) { return t.missed > cfg_.max_missed; }),
                    tracks_.end());

                std::vector<Box> out;
                out.reserve(tracks_.size());
                for (const auto& t : tracks_) {
                    if (t.hits < cfg_.min_hits && t.missed > 0) continue;
                    Box b = t.box;
                    b.id = t.id;
                    b.occluded = t.missed > 0;
                    out.push_back(std::move(b));
                }
                return out;
            }

        private:
            void apply_match_(TrackState& track, const Box& det) {
                const float alpha = 0.5f;
                const float new_vx = det.x - track.box.x;
                const float new_vy = det.y - track.box.y;
                const float new_vw = det.w - track.box.w;
                const float new_vh = det.h - track.box.h;

                track.vx = alpha * new_vx + (1.0f - alpha) * track.vx;
                track.vy = alpha * new_vy + (1.0f - alpha) * track.vy;
                track.vw = alpha * new_vw + (1.0f - alpha) * track.vw;
                track.vh = alpha * new_vh + (1.0f - alpha) * track.vh;

                track.box = det;
                track.hits += 1;
                track.missed = 0;
            }

            void match_greedy_(const std::vector<int>& track_candidates,
                               const std::vector<int>& det_candidates,
                               const std::vector<Box>& detections,
                               float iou_thresh,
                               std::vector<int>& unmatched_tracks,
                               std::vector<int>& unmatched_dets) {
                struct PairScore {
                    int ti = -1;
                    int di = -1;
                    float iou = 0.0f;
                };

                std::vector<PairScore> candidates;
                candidates.reserve(track_candidates.size() * det_candidates.size());
                for (int ti : track_candidates) {
                    for (int di : det_candidates) {
                        const float iou = iou_of(tracks_[static_cast<size_t>(ti)].box,
                                                 detections[static_cast<size_t>(di)]);
                        if (iou >= iou_thresh) {
                            candidates.push_back(PairScore{ti, di, iou});
                        }
                    }
                }

                std::sort(candidates.begin(),
                          candidates.end(),
                          [](const PairScore& a, const PairScore& b) { return a.iou > b.iou; });

                std::vector<char> track_taken(tracks_.size(), 0);
                std::vector<char> det_taken(detections.size(), 0);

                for (const auto& c : candidates) {
                    if (track_taken[static_cast<size_t>(c.ti)] ||
                        det_taken[static_cast<size_t>(c.di)]) {
                        continue;
                    }
                    track_taken[static_cast<size_t>(c.ti)] = 1;
                    det_taken[static_cast<size_t>(c.di)] = 1;
                    apply_match_(tracks_[static_cast<size_t>(c.ti)],
                                 detections[static_cast<size_t>(c.di)]);
                }

                unmatched_tracks.clear();
                unmatched_dets.clear();
                for (int ti : track_candidates) {
                    if (!track_taken[static_cast<size_t>(ti)]) unmatched_tracks.push_back(ti);
                }
                for (int di : det_candidates) {
                    if (!det_taken[static_cast<size_t>(di)]) unmatched_dets.push_back(di);
                }
            }

            TrackerConfig cfg_;
            int next_track_id_ = 1;
            std::vector<TrackState> tracks_;
        };
    } // namespace

    std::unique_ptr<ITracker> create_demo_tracker(const TrackerConfig& cfg) {
        return std::make_unique<DemoTracker>(cfg);
    }
}
