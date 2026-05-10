#include <pipeline/stream_coordinator.hpp>

#include <algorithm>
#include <chrono>
#include <utility>

namespace veilsight {
    StreamCoordinator::StreamCoordinator(std::unique_ptr<ITracker> tracker,
                                         FaceDetectorModuleConfig face_detector,
                                         bool face_detection_enabled,
                                         int64_t reorder_window,
                                         size_t pending_limit)
        : tracker_(std::move(tracker)),
          face_probe_planner_(face_detector),
          face_result_applier_(face_detector, face_probe_planner_.state_store()),
          face_detection_enabled_(face_detection_enabled),
          reorder_window_(std::max<int64_t>(0, reorder_window)),
          pending_limit_(std::max<size_t>(1, pending_limit)) {}

    void StreamCoordinator::push_frame(const FramePtr& frame) {
        if (!frame) return;
        if (next_tracker_frame_id_ >= 0 && frame->frame_id < next_tracker_frame_id_) return;
        pending_frames_[frame->frame_id] = frame;
    }

    void StreamCoordinator::push_person_detection(PersonDetectionResult result) {
        if (next_tracker_frame_id_ >= 0 && result.frame_id < next_tracker_frame_id_) return;
        pending_person_detections_[result.frame_id] = std::move(result);
    }

    void StreamCoordinator::push_face_result(FaceDetectionResult result) {
        auto it = pending_face_frames_.find(result.frame_id);
        if (it == pending_face_frames_.end()) return;
        auto& pending = it->second;
        if (pending.pending_probe_ids.erase(result.probe_id) == 0) return;
        pending.results.push_back(std::move(result));
    }

    void StreamCoordinator::push_recognition_result(RecognitionResult result) {
        if (next_recognition_frame_id_ >= 0 && result.frame_id < next_recognition_frame_id_) return;
        pending_recognitions_[result.frame_id] = std::move(result);
    }

    void StreamCoordinator::push_identity_result(IdentityResult result) {
        if (next_commit_frame_id_ >= 0 && result.frame_id < next_commit_frame_id_) return;
        pending_identities_[result.frame_id] = std::move(result);
    }

    void StreamCoordinator::drain_ready(const Callbacks& callbacks) {
        drain_tracking_(callbacks);
        drain_face_ready_(callbacks);
        drain_recognition_ready_(callbacks);
        drain_commit_ready_(callbacks);
        trim_pending_();
    }

    void StreamCoordinator::drain_tracking_(const Callbacks& callbacks) {
        if (next_tracker_frame_id_ < 0 && !pending_frames_.empty()) {
            next_tracker_frame_id_ = pending_frames_.begin()->first;
        }

        while (next_tracker_frame_id_ >= 0) {
            auto frame_it = pending_frames_.find(next_tracker_frame_id_);
            auto det_it = pending_person_detections_.find(next_tracker_frame_id_);

            if (frame_it != pending_frames_.end() && det_it != pending_person_detections_.end()) {
                process_tracked_frame_(frame_it->second, det_it->second.boxes, callbacks);
                pending_frames_.erase(frame_it);
                pending_person_detections_.erase(det_it);
                ++next_tracker_frame_id_;
                continue;
            }

            if (frame_it == pending_frames_.end()) {
                if (!pending_frames_.empty() && pending_frames_.begin()->first > next_tracker_frame_id_) {
                    next_tracker_frame_id_ = pending_frames_.begin()->first;
                    continue;
                }
                break;
            }

            const int64_t latest_frame = pending_frames_.empty() ? next_tracker_frame_id_ : pending_frames_.rbegin()->first;
            const int64_t latest_det = pending_person_detections_.empty()
                                           ? next_tracker_frame_id_
                                           : pending_person_detections_.rbegin()->first;
            const int64_t latest_seen = std::max(latest_frame, latest_det);
            if (latest_seen - next_tracker_frame_id_ > reorder_window_) {
                process_tracked_frame_(frame_it->second, {}, callbacks);
                pending_frames_.erase(frame_it);
                ++next_tracker_frame_id_;
                continue;
            }
            break;
        }

        while (!pending_person_detections_.empty() &&
               next_tracker_frame_id_ >= 0 &&
               pending_person_detections_.begin()->first < next_tracker_frame_id_) {
            pending_person_detections_.erase(pending_person_detections_.begin());
        }
    }

    void StreamCoordinator::process_tracked_frame_(const FramePtr& frame,
                                                  const std::vector<Box>& detections,
                                                  const Callbacks& callbacks) {
        if (!frame || !tracker_) return;

        const auto tracker_t0 = std::chrono::steady_clock::now();
        frame->tracked_boxes = tracker_->update(
            TrackerFrameInfo{
                frame->stream_id,
                frame->frame_id,
                frame->inf_w,
                frame->inf_h,
            },
            detections);
        const auto tracker_t1 = std::chrono::steady_clock::now();

        if (callbacks.on_tracker_timing) {
            callbacks.on_tracker_timing(
                *frame,
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(tracker_t1 - tracker_t0).count()));
        }

        PendingFaceFrame pending;
        pending.frame = frame;
        pending.tracks = frame->tracked_boxes;

        std::vector<FaceDetectionTask> probes;
        if (face_detection_enabled_) {
            probes = face_probe_planner_.plan(*frame, pending.tracks);
        } else {
            for (auto& track : pending.tracks) {
                track.privacy_action = "anonymize";
                track.face.reset();
            }
        }

        if (next_commit_frame_id_ < 0) {
            next_commit_frame_id_ = frame->frame_id;
        }

        if (probes.empty()) {
            pending_face_frames_[frame->frame_id] = std::move(pending);
            auto it = pending_face_frames_.find(frame->frame_id);
            if (it != pending_face_frames_.end()) {
                queue_recognition_(it->second, callbacks);
            }
            return;
        }

        for (const auto& probe : probes) {
            pending.pending_probe_ids.insert(probe.probe_id);
        }
        latest_face_queued_frame_id_ = std::max(latest_face_queued_frame_id_, frame->frame_id);
        pending_face_frames_[frame->frame_id] = std::move(pending);
        if (callbacks.on_face_probes_ready) {
            callbacks.on_face_probes_ready(std::move(probes));
        }
    }

    void StreamCoordinator::drain_face_ready_(const Callbacks& callbacks) {
        for (auto it = pending_face_frames_.begin(); it != pending_face_frames_.end();) {
            auto& pending = it->second;
            const bool timed_out =
                !pending.pending_probe_ids.empty() &&
                latest_face_queued_frame_id_ >= 0 &&
                latest_face_queued_frame_id_ - it->first > reorder_window_;
            if (!pending.recognition_queued &&
                (pending.pending_probe_ids.empty() || timed_out)) {
                pending.pending_probe_ids.clear();
                queue_recognition_(pending, callbacks);
            }

            if (pending.recognition_queued) {
                it = pending_face_frames_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void StreamCoordinator::queue_recognition_(PendingFaceFrame& pending, const Callbacks& callbacks) {
        if (!pending.frame || pending.recognition_queued) return;
        size_t face_count = 0;
        for (const auto& result : pending.results) {
            face_count += result.faces.size();
        }
        pending.frame->face_detection_count = face_count;
        face_result_applier_.apply(*pending.frame, pending.tracks, pending.results);
        pending.frame->tracked_boxes = pending.tracks;

        RecognitionTask task;
        task.stream_id = pending.frame->stream_id;
        task.frame_id = pending.frame->frame_id;
        task.frame = pending.frame;
        task.tracks = pending.tracks;
        latest_recognition_queued_frame_id_ = std::max(latest_recognition_queued_frame_id_, task.frame_id);
        pending.recognition_queued = true;
        if (callbacks.on_recognition_ready) {
            callbacks.on_recognition_ready(std::move(task));
        }
    }

    void StreamCoordinator::drain_recognition_ready_(const Callbacks& callbacks) {
        if (next_recognition_frame_id_ < 0 && !pending_recognitions_.empty()) {
            next_recognition_frame_id_ = pending_recognitions_.begin()->first;
        }

        while (next_recognition_frame_id_ >= 0) {
            auto it = pending_recognitions_.find(next_recognition_frame_id_);
            if (it != pending_recognitions_.end()) {
                queue_identity_(it->second, callbacks);
                pending_recognitions_.erase(it);
                ++next_recognition_frame_id_;
                continue;
            }

            if (latest_recognition_queued_frame_id_ >= 0 &&
                latest_recognition_queued_frame_id_ - next_recognition_frame_id_ > reorder_window_) {
                ++next_recognition_frame_id_;
                continue;
            }
            break;
        }

        while (!pending_recognitions_.empty() &&
               next_recognition_frame_id_ >= 0 &&
               pending_recognitions_.begin()->first < next_recognition_frame_id_) {
            pending_recognitions_.erase(pending_recognitions_.begin());
        }
    }

    void StreamCoordinator::queue_identity_(RecognitionResult& result, const Callbacks& callbacks) {
        if (!result.frame) return;
        result.frame->tracked_boxes = result.tracks;

        IdentityTask task;
        task.stream_id = result.stream_id;
        task.frame_id = result.frame_id;
        task.frame = result.frame;
        task.tracks = result.tracks;
        latest_identity_queued_frame_id_ = std::max(latest_identity_queued_frame_id_, task.frame_id);
        if (callbacks.on_identity_ready) {
            callbacks.on_identity_ready(std::move(task));
        }
    }

    void StreamCoordinator::drain_commit_ready_(const Callbacks& callbacks) {
        if (next_commit_frame_id_ < 0 && !pending_identities_.empty()) {
            next_commit_frame_id_ = pending_identities_.begin()->first;
        }

        while (next_commit_frame_id_ >= 0) {
            auto it = pending_identities_.find(next_commit_frame_id_);
            if (it != pending_identities_.end()) {
                if (it->second.frame) {
                    it->second.frame->tracked_boxes = it->second.tracks;
                    if (callbacks.on_frame_committed) callbacks.on_frame_committed(it->second.frame);
                }
                pending_identities_.erase(it);
                ++next_commit_frame_id_;
                continue;
            }

            if (latest_identity_queued_frame_id_ >= 0 &&
                latest_identity_queued_frame_id_ - next_commit_frame_id_ > reorder_window_) {
                ++next_commit_frame_id_;
                continue;
            }
            break;
        }

        while (!pending_identities_.empty() &&
               next_commit_frame_id_ >= 0 &&
               pending_identities_.begin()->first < next_commit_frame_id_) {
            pending_identities_.erase(pending_identities_.begin());
        }
    }

    void StreamCoordinator::trim_pending_() {
        while (pending_frames_.size() > pending_limit_) {
            pending_frames_.erase(pending_frames_.begin());
        }
        while (pending_person_detections_.size() > pending_limit_) {
            pending_person_detections_.erase(pending_person_detections_.begin());
        }
        while (pending_face_frames_.size() > pending_limit_) {
            auto it = pending_face_frames_.begin();
            if (next_commit_frame_id_ < 0 || it->first >= next_commit_frame_id_) {
                next_commit_frame_id_ = it->first + 1;
            }
            pending_face_frames_.erase(it);
        }
        while (pending_recognitions_.size() > pending_limit_) {
            pending_recognitions_.erase(pending_recognitions_.begin());
        }
        while (pending_identities_.size() > pending_limit_) {
            pending_identities_.erase(pending_identities_.begin());
        }
    }
}
