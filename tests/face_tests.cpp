#include <face_detector/face_policy.hpp>
#include <recognizer/recognizer.hpp>
#include <face_detector/scrfd_detector.hpp>

#include <cmath>
#include <iostream>
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

    bool near(float a, float b, float eps = 0.001f) {
        return std::fabs(a - b) <= eps;
    }

    veilsight::Box person(float x, float y, float w, float h, int id) {
        veilsight::Box b;
        b.x = x;
        b.y = y;
        b.w = w;
        b.h = h;
        b.id = id;
        b.score = 0.9f;
        return b;
    }

    veilsight::FaceObservation face(float x, float y, float w, float h, float score = 0.9f) {
        veilsight::FaceObservation f;
        f.bbox.x = x;
        f.bbox.y = y;
        f.bbox.w = w;
        f.bbox.h = h;
        f.score = score;
        return f;
    }

    veilsight::FrameCtx frame(int64_t frame_id) {
        veilsight::FrameCtx f;
        f.stream_id = "cam0";
        f.frame_id = frame_id;
        f.inf_w = 640;
        f.inf_h = 480;
        f.inf = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
        return f;
    }

    class FakeFaceDetector final : public veilsight::IFaceDetector {
    public:
        std::vector<std::vector<veilsight::FaceObservation>> responses;
        std::vector<veilsight::FaceDetectorRunConfig> runs;

        std::vector<veilsight::FaceObservation> detect_faces(
            const cv::Mat&,
            const veilsight::FaceDetectorRunConfig& run) override {
            runs.push_back(run);
            if (responses.empty()) return {};

            auto out = responses.front();
            responses.erase(responses.begin());
            return out;
        }
    };

    veilsight::FacePolicyConfig test_policy() {
        veilsight::FacePolicyConfig cfg;
        cfg.full_frame_interval = 1000;
        cfg.full_frame_input_w = 640;
        cfg.full_frame_input_h = 640;
        cfg.roi_input_w = 320;
        cfg.roi_input_h = 320;
        cfg.max_roi_probes_per_frame = 2;
        cfg.refresh_interval = 15;
        cfg.reuse_ttl = 45;
        cfg.miss_retry_initial = 5;
        cfg.miss_retry_max = 30;
        cfg.min_track_height = 48;
        cfg.min_face_score = 0.45f;
        return cfg;
    }

    void test_scrfd_500m_landmarks_loads_and_runs_on_blank_image() {
        veilsight::SCRFDModuleConfig cfg;
        cfg.variant = "500m_landmarks";
        cfg.param_path = "models/face_detectors/scrfd/500m/scrfd_500m_l.ncnn.param";
        cfg.bin_path = "models/face_detectors/scrfd/500m/scrfd_500m_l.ncnn.bin";
        cfg.score_threshold = 0.45f;
        cfg.top_k = 10;
        cfg.ncnn_threads = 1;

        veilsight::SCRFDDetector detector(cfg);
        const cv::Mat blank(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));
        const auto faces = detector.detect_faces(blank, veilsight::FaceDetectorRunConfig{640, 640});

        for (const auto& f : faces) {
            check(f.bbox.x >= 0.0f && f.bbox.y >= 0.0f, "SCRFD face boxes should be clipped to origin");
            check(f.bbox.x + f.bbox.w <= 320.1f && f.bbox.y + f.bbox.h <= 240.1f,
                  "SCRFD face boxes should be clipped to image bounds");
        }
    }

    void test_scrfd_landmark_outputs_decode_when_candidates_exist() {
        veilsight::SCRFDModuleConfig cfg;
        cfg.variant = "500m_landmarks";
        cfg.param_path = "models/face_detectors/scrfd/500m/scrfd_500m_l.ncnn.param";
        cfg.bin_path = "models/face_detectors/scrfd/500m/scrfd_500m_l.ncnn.bin";
        cfg.score_threshold = -1.0f;
        cfg.nms_threshold = 0.30f;
        cfg.top_k = 1;
        cfg.ncnn_threads = 1;

        veilsight::SCRFDDetector detector(cfg);
        const cv::Mat blank(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));
        const auto faces = detector.detect_faces(blank, veilsight::FaceDetectorRunConfig{640, 640});

        check(!faces.empty(), "SCRFD landmark model should emit at least one low-threshold candidate on blank input");
        if (!faces.empty()) {
            check(faces[0].landmark_count == 5, "SCRFD landmark-capable model should decode five landmarks");
        }
    }

    void test_roi_mapping_offsets_face_and_landmarks() {
        auto f = face(10.0f, 20.0f, 30.0f, 40.0f);
        f.landmark_count = 2;
        f.landmarks[0] = veilsight::PointF{11.0f, 21.0f};
        f.landmarks[1] = veilsight::PointF{12.0f, 22.0f};

        const auto mapped = veilsight::map_face_from_roi(f, cv::Rect(100, 50, 120, 160), 42,
                                                         veilsight::FaceSource::PersonRoi);
        check(near(mapped.bbox.x, 110.0f) && near(mapped.bbox.y, 70.0f),
              "ROI mapping should offset face bbox into inference-frame coordinates");
        check(near(mapped.landmarks[0].x, 111.0f) && near(mapped.landmarks[0].y, 71.0f),
              "ROI mapping should offset landmarks into inference-frame coordinates");
        check(mapped.frame_id == 42 && mapped.source == veilsight::FaceSource::PersonRoi && mapped.fresh,
              "ROI mapping should mark mapped observations fresh and sourced from person ROI");
    }

    void test_identity_transform_maps_geometry_unchanged() {
        const auto transform = veilsight::identity_transform(cv::Size(320, 240));
        const auto r = veilsight::map_rect(transform, veilsight::RectF{10.0f, 20.0f, 30.0f, 40.0f});
        const auto p = veilsight::map_point(transform, veilsight::PointF{11.0f, 22.0f});

        check(near(r.x, 10.0f) && near(r.y, 20.0f) && near(r.w, 30.0f) && near(r.h, 40.0f),
              "identity transform should leave rectangles unchanged");
        check(near(p.x, 11.0f) && near(p.y, 22.0f),
              "identity transform should leave points unchanged");
    }

    void test_roi_translation_transform_maps_face_to_frame() {
        veilsight::SpatialTransform transform = veilsight::identity_transform(cv::Size(120, 160));
        transform.target_size = cv::Size(640, 480);
        transform.source_to_target = cv::Matx23f(1.0f, 0.0f, 100.0f,
                                                 0.0f, 1.0f, 50.0f);
        transform.target_to_source = cv::Matx23f(1.0f, 0.0f, -100.0f,
                                                 0.0f, 1.0f, -50.0f);

        auto f = face(10.0f, 20.0f, 30.0f, 40.0f);
        f.landmark_count = 1;
        f.landmarks[0] = veilsight::PointF{12.0f, 22.0f};
        const auto mapped = veilsight::map_face(transform, f);

        check(near(mapped.bbox.x, 110.0f) && near(mapped.bbox.y, 70.0f),
              "ROI transform should translate local face bbox into frame coordinates");
        check(near(mapped.landmarks[0].x, 112.0f) && near(mapped.landmarks[0].y, 72.0f),
              "ROI transform should translate local landmarks into frame coordinates");
    }

    void test_full_frame_assignment_chooses_correct_track() {
        auto cfg = test_policy();
        cfg.full_frame_interval = 1;
        cfg.max_roi_probes_per_frame = 0;
        FakeFaceDetector detector;
        detector.responses.push_back({face(410.0f, 55.0f, 48.0f, 48.0f)});

        auto state = std::make_shared<veilsight::FaceStateStore>();
        veilsight::HybridFacePolicy policy(cfg, state);
        auto f = frame(30);
        std::vector<veilsight::Box> tracks = {
            person(80.0f, 50.0f, 120.0f, 260.0f, 1),
            person(380.0f, 45.0f, 125.0f, 270.0f, 2),
        };

        policy.annotate(f, tracks, detector);

        check(!tracks[0].face.has_value(), "full-frame face should not attach to the wrong person track");
        check(tracks[1].face.has_value(), "full-frame face should attach to containing upper-body track");
        check(tracks[1].face && tracks[1].face->source == veilsight::FaceSource::FullFrame,
              "assigned full-frame face should keep FullFrame source");
    }

    void test_roi_scheduler_respects_probe_budget() {
        auto cfg = test_policy();
        cfg.max_roi_probes_per_frame = 1;
        FakeFaceDetector detector;
        auto state = std::make_shared<veilsight::FaceStateStore>();
        state->streams["cam0"].last_full_frame_sweep = 0;
        veilsight::HybridFacePolicy policy(cfg, state);

        auto f = frame(1);
        std::vector<veilsight::Box> tracks = {
            person(20.0f, 20.0f, 80.0f, 180.0f, 1),
            person(180.0f, 20.0f, 80.0f, 180.0f, 2),
            person(340.0f, 20.0f, 80.0f, 180.0f, 3),
        };

        policy.annotate(f, tracks, detector);

        check(detector.runs.size() == 1, "ROI scheduler should cap detector calls at max_roi_probes_per_frame");
        check(detector.runs.empty() || detector.runs[0].input_w == 320,
              "ROI scheduler should use ROI detector input size");
    }

    void test_cache_reuse_predicts_face_from_relative_coordinates() {
        veilsight::FaceTrackCacheEntry entry;
        entry.face_rel = veilsight::RectF{0.25f, 0.10f, 0.50f, 0.20f};
        entry.has_face_rel = true;
        entry.last_face = face(0.0f, 0.0f, 50.0f, 50.0f, 0.88f);
        entry.last_face_frame = 10;

        const auto p = person(100.0f, 50.0f, 120.0f, 240.0f, 7);
        const auto predicted = veilsight::predict_face_from_cache(entry, p, 11);

        check(predicted.has_value(), "cache reuse should predict a face when relative coordinates are available");
        check(predicted && near(predicted->bbox.x, 130.0f) && near(predicted->bbox.y, 74.0f),
              "cache reuse should map relative face origin through the current person box");
        check(predicted && near(predicted->bbox.w, 60.0f) && near(predicted->bbox.h, 48.0f),
              "cache reuse should map relative face size through the current person box");
        check(predicted && predicted->source == veilsight::FaceSource::Predicted && !predicted->fresh,
              "cache reuse should mark predicted observations as stale/non-fresh");
    }

    void test_miss_backoff_delays_repeated_roi_probes() {
        auto cfg = test_policy();
        cfg.max_roi_probes_per_frame = 1;
        cfg.miss_retry_initial = 5;
        FakeFaceDetector detector;
        auto state = std::make_shared<veilsight::FaceStateStore>();
        state->streams["cam0"].last_full_frame_sweep = 0;
        veilsight::HybridFacePolicy policy(cfg, state);

        auto f1 = frame(1);
        std::vector<veilsight::Box> tracks1 = {person(120.0f, 40.0f, 100.0f, 220.0f, 10)};
        policy.annotate(f1, tracks1, detector);
        const size_t after_first = detector.runs.size();

        auto f2 = frame(2);
        std::vector<veilsight::Box> tracks2 = {person(122.0f, 40.0f, 100.0f, 220.0f, 10)};
        policy.annotate(f2, tracks2, detector);

        check(after_first == 1, "first missing-face track should receive one ROI probe");
        check(detector.runs.size() == after_first, "miss backoff should delay repeated ROI probes");
        check(tracks2[0].privacy_action == "anonymize", "no-face fallback should keep whole person anonymized");
    }

    void test_noop_recognizer_passes_tracks_through() {
        veilsight::RecognizerModuleConfig cfg;
        cfg.type = "noop";

        const auto factory = veilsight::create_recognizer_factory(cfg);
        check(factory->backend_threads() == 1, "noop recognizer should report one backend thread");

        auto recognizer = factory->create();
        veilsight::RecognitionTask task;
        task.stream_id = "cam0";
        task.frame_id = 42;
        task.tracks = {person(10.0f, 20.0f, 30.0f, 40.0f, 7)};

        const auto result = recognizer->recognize(task);
        check(result.stream_id == "cam0" && result.frame_id == 42,
              "noop recognizer should preserve task identity");
        check(result.tracks.size() == 1 && result.tracks[0].id == 7,
              "noop recognizer should pass tracks through unchanged");
    }
}

int main() {
    test_scrfd_500m_landmarks_loads_and_runs_on_blank_image();
    test_scrfd_landmark_outputs_decode_when_candidates_exist();
    test_roi_mapping_offsets_face_and_landmarks();
    test_identity_transform_maps_geometry_unchanged();
    test_roi_translation_transform_maps_face_to_frame();
    test_full_frame_assignment_chooses_correct_track();
    test_roi_scheduler_respects_probe_budget();
    test_cache_reuse_predicts_face_from_relative_coordinates();
    test_miss_backoff_delays_repeated_roi_probes();
    test_noop_recognizer_passes_tracks_through();

    if (g_failures != 0) {
        std::cerr << "[FAIL] total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "[OK] all face tests passed\n";
    return 0;
}
