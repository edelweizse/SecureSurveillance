#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include <common/config.hpp>
#include <face_detector/face_policy.hpp>
#include <pipeline/tasks.hpp>
#include <tracking/tracker.hpp>

namespace veilsight {
    class StreamCoordinator {
    public:
        struct Callbacks {
            std::function<void(const FrameCtx&, uint64_t)> on_tracker_timing;
            std::function<void(std::vector<FaceDetectionTask>)> on_face_probes_ready;
            std::function<void(RecognitionTask)> on_recognition_ready;
            std::function<void(IdentityTask)> on_identity_ready;
            std::function<void(const FramePtr&)> on_frame_committed;
        };

        StreamCoordinator(std::unique_ptr<ITracker> tracker,
                          FaceDetectorModuleConfig face_detector,
                          bool face_detection_enabled,
                          int64_t reorder_window = 5,
                          size_t pending_limit = 500);

        void push_frame(const FramePtr& frame);
        void push_person_detection(PersonDetectionResult result);
        void push_face_result(FaceDetectionResult result);
        void push_recognition_result(RecognitionResult result);
        void push_identity_result(IdentityResult result);
        void drain_ready(const Callbacks& callbacks);

    private:
        struct PendingFaceFrame {
            FramePtr frame;
            std::vector<Box> tracks;
            std::set<std::string> pending_probe_ids;
            std::vector<FaceDetectionResult> results;
            bool recognition_queued = false;
        };

        void drain_tracking_(const Callbacks& callbacks);
        void drain_independent_face_probes_(const Callbacks& callbacks);
        void process_tracked_frame_(const FramePtr& frame,
                                    const std::vector<Box>& detections,
                                    const Callbacks& callbacks);
        void drain_face_ready_(const Callbacks& callbacks);
        void queue_recognition_(PendingFaceFrame& pending, const Callbacks& callbacks);
        void drain_recognition_ready_(const Callbacks& callbacks);
        void queue_identity_(RecognitionResult& result, const Callbacks& callbacks);
        void drain_commit_ready_(const Callbacks& callbacks);
        void trim_pending_();

        std::unique_ptr<ITracker> tracker_;
        FaceProbePlanner face_probe_planner_;
        FaceResultApplier face_result_applier_;
        bool face_detection_enabled_ = false;
        bool independent_face_detection_ = false;
        int64_t reorder_window_ = 5;
        size_t pending_limit_ = 500;

        int64_t next_tracker_frame_id_ = -1;
        int64_t next_recognition_frame_id_ = -1;
        int64_t next_commit_frame_id_ = -1;
        int64_t latest_face_queued_frame_id_ = -1;
        int64_t latest_recognition_queued_frame_id_ = -1;
        int64_t latest_identity_queued_frame_id_ = -1;

        std::map<int64_t, FramePtr> pending_frames_;
        std::map<int64_t, PersonDetectionResult> pending_person_detections_;
        std::map<int64_t, std::set<std::string>> queued_independent_face_probe_ids_;
        std::map<int64_t, std::vector<FaceDetectionResult>> early_face_results_;
        std::map<int64_t, PendingFaceFrame> pending_face_frames_;
        std::map<int64_t, RecognitionResult> pending_recognitions_;
        std::map<int64_t, IdentityResult> pending_identities_;
    };
}
