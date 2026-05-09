#include <anonymization/anonymizer.hpp>
#include <face_detector/face_policy.hpp>
#include <identity/identity_decider.hpp>
#include <pipeline/bounded_queue.hpp>
#include <pipeline/metrics.hpp>
#include <pipeline/stream_coordinator.hpp>
#include <tracking/tracker.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace {
    int g_failures = 0;

    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++g_failures;
            std::cerr << "[FAIL] " << message << "\n";
        }
    }

    veilsight::Box box(float x = 10.0f, float y = 10.0f, float w = 20.0f, float h = 40.0f) {
        veilsight::Box b;
        b.x = x;
        b.y = y;
        b.w = w;
        b.h = h;
        b.score = 0.9f;
        return b;
    }

    veilsight::FramePtr frame(int64_t frame_id) {
        auto f = std::make_shared<veilsight::FrameCtx>();
        f->stream_id = "cam0";
        f->frame_id = frame_id;
        f->inf_w = 120;
        f->inf_h = 100;
        f->ui_w = 120;
        f->ui_h = 100;
        f->inf = cv::Mat(100, 120, CV_8UC3, cv::Scalar(0, 0, 0));
        f->ui = cv::Mat(100, 120, CV_8UC3, cv::Scalar(0, 0, 0));
        return f;
    }

    class EchoTracker final : public veilsight::ITracker {
    public:
        std::vector<veilsight::Box> update(const veilsight::TrackerFrameInfo& info,
                                           const std::vector<veilsight::Box>& detections) override {
            std::vector<veilsight::Box> out = detections;
            for (auto& item : out) {
                item.id = static_cast<int>(info.frame_id);
                item.privacy_action = "anonymize";
            }
            return out;
        }
    };

    void attach_noop_identity(veilsight::StreamCoordinator& coordinator,
                              veilsight::StreamCoordinator::Callbacks& callbacks) {
        callbacks.on_recognition_ready = [&coordinator](veilsight::RecognitionTask task) {
            veilsight::RecognitionResult result;
            result.stream_id = task.stream_id;
            result.frame_id = task.frame_id;
            result.frame = task.frame;
            result.tracks = task.tracks;
            coordinator.push_recognition_result(std::move(result));
        };
        callbacks.on_identity_ready = [&coordinator](veilsight::IdentityTask task) {
            veilsight::IdentityResult result;
            result.stream_id = task.stream_id;
            result.frame_id = task.frame_id;
            result.frame = task.frame;
            result.tracks = task.tracks;
            for (auto& track : result.tracks) {
                track.privacy_action = "anonymize";
            }
            coordinator.push_identity_result(std::move(result));
        };
    }

    void test_queue_catalog_snapshot_has_new_names_and_metadata() {
        veilsight::NamedQueue<veilsight::PersonDetectionTask> queue(
            "global/person_detector.in",
            "Ingestor",
            "PersonDetectorPool",
            "Shared people detector work from all streams.",
            8);
        const auto snapshot = queue.snapshot();
        check(queue.name() == "global/person_detector.in", "queue catalog should use global person detector name");
        check(snapshot.producer == "Ingestor", "queue snapshot should carry producer metadata");
        check(snapshot.consumer == "PersonDetectorPool", "queue snapshot should carry consumer metadata");

        std::map<std::string, veilsight::QueueSnapshot> queues;
        queues[queue.name()] = snapshot;
        queues["global/recognizer.in"] = veilsight::QueueSnapshot{0, 8, 0, "StreamCoordinator", "RecognizerPool", "Recognition tasks"};
        queues["stream/cam0/frames.in"] = veilsight::QueueSnapshot{0, 4, 0, "Ingestor", "StreamCoordinator", "FrameCtx"};
        const auto json = veilsight::metrics_snapshot_to_json(veilsight::RuntimeMetrics::Snapshot{}, queues);
        check(json.find("global/person_detector.in") != std::string::npos, "metrics JSON should contain new queue names");
        check(json.find("global/recognizer.in") != std::string::npos, "metrics JSON should contain recognizer queue name");
        check(json.find("analytics_out") == std::string::npos, "metrics JSON should not contain analytics_out");
        check(json.find("producer") != std::string::npos, "metrics JSON should keep queue metadata");
    }

    void test_stream_coordinator_commits_in_order_with_out_of_order_person_detections() {
        veilsight::FaceDetectorModuleConfig face_detector;
        veilsight::StreamCoordinator coordinator(
            std::make_unique<EchoTracker>(),
            face_detector,
            false,
            5,
            32);

        std::vector<int64_t> commits;
        veilsight::StreamCoordinator::Callbacks callbacks;
        attach_noop_identity(coordinator, callbacks);
        callbacks.on_frame_committed = [&commits](const veilsight::FramePtr& f) {
            commits.push_back(f->frame_id);
        };

        coordinator.push_frame(frame(1));
        coordinator.push_frame(frame(2));
        coordinator.push_person_detection(veilsight::PersonDetectionResult{"cam0", 2, {box(30, 10, 20, 40)}});
        coordinator.drain_ready(callbacks);
        check(commits.empty(), "coordinator should wait for frame 1 before committing frame 2");

        coordinator.push_person_detection(veilsight::PersonDetectionResult{"cam0", 1, {box(10, 10, 20, 40)}});
        coordinator.drain_ready(callbacks);
        check((commits == std::vector<int64_t>{1, 2}), "coordinator should commit out-of-order detections by frame_id");
    }

    void test_stale_results_are_discarded_after_commit() {
        veilsight::FaceDetectorModuleConfig face_detector;
        veilsight::StreamCoordinator coordinator(
            std::make_unique<EchoTracker>(),
            face_detector,
            false,
            1,
            32);

        std::vector<int64_t> commits;
        veilsight::StreamCoordinator::Callbacks callbacks;
        attach_noop_identity(coordinator, callbacks);
        callbacks.on_frame_committed = [&commits](const veilsight::FramePtr& f) {
            commits.push_back(f->frame_id);
        };

        coordinator.push_frame(frame(1));
        coordinator.push_person_detection(veilsight::PersonDetectionResult{"cam0", 1, {box()}});
        coordinator.drain_ready(callbacks);
        check((commits == std::vector<int64_t>{1}), "frame 1 should commit once");

        coordinator.push_person_detection(veilsight::PersonDetectionResult{"cam0", 1, {box()}});
        coordinator.push_face_result(veilsight::FaceDetectionResult{"cam0", 1, "late", veilsight::FaceProbeKind::FullFrame});
        coordinator.push_recognition_result(veilsight::RecognitionResult{"cam0", 1, frame(1), {box()}});
        coordinator.push_identity_result(veilsight::IdentityResult{"cam0", 1, frame(1), {box()}});
        coordinator.drain_ready(callbacks);
        check((commits == std::vector<int64_t>{1}),
              "late detector/face/recognition/identity results should not recommit stale frames");
    }

    void test_stream_coordinator_orders_face_recognition_identity() {
        veilsight::FaceDetectorModuleConfig face_detector;
        veilsight::StreamCoordinator coordinator(
            std::make_unique<EchoTracker>(),
            face_detector,
            true,
            5,
            32);

        std::vector<std::string> events;
        veilsight::StreamCoordinator::Callbacks callbacks;
        callbacks.on_face_probes_ready = [&events, &coordinator](std::vector<veilsight::FaceDetectionTask> probes) {
            events.push_back("face");
            for (const auto& probe : probes) {
                veilsight::FaceDetectionResult result;
                result.stream_id = probe.stream_id;
                result.frame_id = probe.frame_id;
                result.probe_id = probe.probe_id;
                result.kind = probe.kind;
                coordinator.push_face_result(std::move(result));
            }
        };
        callbacks.on_recognition_ready = [&events, &coordinator](veilsight::RecognitionTask task) {
            events.push_back("recognition");
            veilsight::RecognitionResult result;
            result.stream_id = task.stream_id;
            result.frame_id = task.frame_id;
            result.frame = task.frame;
            result.tracks = task.tracks;
            coordinator.push_recognition_result(std::move(result));
        };
        callbacks.on_identity_ready = [&events, &coordinator](veilsight::IdentityTask task) {
            events.push_back("identity");
            veilsight::IdentityResult result;
            result.stream_id = task.stream_id;
            result.frame_id = task.frame_id;
            result.frame = task.frame;
            result.tracks = task.tracks;
            coordinator.push_identity_result(std::move(result));
        };
        callbacks.on_frame_committed = [&events](const veilsight::FramePtr&) {
            events.push_back("commit");
        };

        coordinator.push_frame(frame(10));
        coordinator.push_person_detection(veilsight::PersonDetectionResult{"cam0", 10, {box(20, 10, 30, 70)}});
        coordinator.drain_ready(callbacks);
        coordinator.drain_ready(callbacks);

        check((events == std::vector<std::string>{"face", "recognition", "identity", "commit"}),
              "coordinator should order face, recognizer, identity, and commit callbacks");
    }

    void test_anonymizer_skips_non_anonymize_boxes() {
        cv::Mat image(24, 48, CV_8UC3);
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uchar>(x * 3),
                                                      static_cast<uchar>(y * 7),
                                                      static_cast<uchar>((x + y) * 2));
            }
        }
        const cv::Mat original = image.clone();

        veilsight::Box anonymize = box(4, 4, 10, 10);
        anonymize.privacy_action = "anonymize";
        veilsight::Box allow = box(30, 4, 10, 10);
        allow.privacy_action = "allow";

        veilsight::Anonymizer anonymizer(veilsight::AnonymizerConfig{});
        anonymizer.apply(image, {anonymize, allow}, 1.0f, 1.0f, 0.0f, 0.0f);

        const cv::Rect left(2, 2, 14, 14);
        const cv::Rect right(28, 2, 14, 14);
        check(cv::norm(image(left), original(left), cv::NORM_L1) > 0.0,
              "anonymize privacy_action should mutate the anonymized ROI");
        check(cv::norm(image(right), original(right), cv::NORM_L1) == 0.0,
              "non-anonymize privacy_action should leave the ROI untouched");
    }

    void test_face_probe_planner_emits_one_full_frame_probe() {
        veilsight::FaceDetectorModuleConfig cfg;
        veilsight::FaceProbePlanner planner(cfg);

        auto f = frame(10);
        std::vector<veilsight::Box> tracks = {box(20, 10, 30, 70)};
        tracks[0].id = 7;
        const auto probes = planner.plan(*f, tracks);

        check(probes.size() == 1, "face probe planner should emit one probe per frame");
        check(!probes.empty() && probes[0].kind == veilsight::FaceProbeKind::FullFrame,
              "face probe planner should emit full-frame probes only");
    }

    void test_passthrough_identity_preserves_recognizer_decisions() {
        veilsight::IdentityModuleConfig cfg;
        cfg.type = "passthrough";
        const auto factory = veilsight::create_identity_decider_factory(cfg);
        check(factory->backend_threads() == 1, "passthrough identity should report one backend thread");

        auto decider = factory->create();
        veilsight::IdentityTask task;
        task.stream_id = "cam0";
        task.frame_id = 12;
        task.frame = frame(12);

        veilsight::Box known = box();
        known.id = 1;
        known.identity_key = "alice";
        known.identity_confidence = 0.88f;
        known.privacy_action = "allow";

        veilsight::Box unknown = box(40, 10, 20, 40);
        unknown.id = 2;
        unknown.identity_confidence = 0.12f;
        unknown.privacy_action = "allow";
        task.tracks = {known, unknown};

        const auto result = decider->decide(task);
        check(result.tracks.size() == 2, "passthrough identity should preserve track count");
        check(result.tracks[0].identity_key == "alice" &&
                  result.tracks[0].identity_confidence == 0.88f &&
                  result.tracks[0].privacy_action == "allow",
              "passthrough identity should preserve known recognizer decisions");
        check(result.tracks[1].identity_key.empty() &&
                  result.tracks[1].identity_confidence == 0.12f &&
                  result.tracks[1].privacy_action == "anonymize",
              "passthrough identity should anonymize empty identities");
        check(task.frame->tracked_boxes.size() == 2 &&
                  task.frame->tracked_boxes[0].identity_key == "alice" &&
                  task.frame->tracked_boxes[1].privacy_action == "anonymize",
              "passthrough identity should update frame tracks");
    }

    void test_noop_identity_still_anonymizes_all_tracks() {
        veilsight::IdentityModuleConfig cfg;
        cfg.type = "noop";
        auto decider = veilsight::create_identity_decider(cfg);
        veilsight::IdentityTask task;
        task.stream_id = "cam0";
        task.frame_id = 13;
        task.frame = frame(13);
        veilsight::Box known = box();
        known.id = 1;
        known.identity_key = "alice";
        known.identity_confidence = 0.91f;
        known.privacy_action = "allow";
        task.tracks = {known};

        const auto result = decider->decide(task);
        check(result.tracks.size() == 1 &&
                  result.tracks[0].identity_key.empty() &&
                  result.tracks[0].identity_confidence == 0.0f &&
                  result.tracks[0].privacy_action == "anonymize",
              "noop identity should continue anonymizing all tracks");
    }
}

int main() {
    test_queue_catalog_snapshot_has_new_names_and_metadata();
    test_stream_coordinator_commits_in_order_with_out_of_order_person_detections();
    test_stale_results_are_discarded_after_commit();
    test_stream_coordinator_orders_face_recognition_identity();
    test_anonymizer_skips_non_anonymize_boxes();
    test_face_probe_planner_emits_one_full_frame_probe();
    test_passthrough_identity_preserves_recognizer_decisions();
    test_noop_identity_still_anonymizes_all_tracks();

    if (g_failures != 0) {
        std::cerr << "[FAIL] total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "[OK] all pipeline tests passed\n";
    return 0;
}
