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
        f.source_type = "file";
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

    veilsight::FaceDetectorModuleConfig test_face_detector_config() {
        veilsight::FaceDetectorModuleConfig cfg;
        cfg.type = "scrfd";
        cfg.scrfd.input_w = 640;
        cfg.scrfd.input_h = 640;
        cfg.scrfd.score_threshold = 0.45f;
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

    void test_identity_transform_maps_geometry_unchanged() {
        const auto transform = veilsight::identity_transform(cv::Size(320, 240));
        const auto r = veilsight::map_rect(transform, veilsight::RectF{10.0f, 20.0f, 30.0f, 40.0f});
        const auto p = veilsight::map_point(transform, veilsight::PointF{11.0f, 22.0f});

        check(near(r.x, 10.0f) && near(r.y, 20.0f) && near(r.w, 30.0f) && near(r.h, 40.0f),
              "identity transform should leave rectangles unchanged");
        check(near(p.x, 11.0f) && near(p.y, 22.0f),
              "identity transform should leave points unchanged");
    }

    void test_translation_transform_maps_face_to_frame() {
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
              "translation transform should translate local face bbox into frame coordinates");
        check(near(mapped.landmarks[0].x, 112.0f) && near(mapped.landmarks[0].y, 72.0f),
              "translation transform should translate local landmarks into frame coordinates");
    }

    void test_full_frame_assignment_chooses_correct_track() {
        auto cfg = test_face_detector_config();
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

    void test_unassigned_webcam_faces_become_face_only_boxes() {
        auto cfg = test_face_detector_config();
        FakeFaceDetector detector;
        detector.responses.push_back({
            face(140.0f, 70.0f, 48.0f, 48.0f),
            face(320.0f, 80.0f, 44.0f, 44.0f),
        });

        auto state = std::make_shared<veilsight::FaceStateStore>();
        veilsight::HybridFacePolicy policy(cfg, state);
        auto f = frame(31);
        f.source_type = "webcam";
        std::vector<veilsight::Box> tracks;

        policy.annotate(f, tracks, detector);

        check(tracks.size() == 2, "unassigned webcam faces should be emitted as face-only boxes");
        check(tracks[0].id < 0 && tracks[1].id < 0, "face-only boxes should use synthetic negative IDs");
        check(tracks[0].privacy_action == "anonymize", "face-only boxes should be anonymized");
        check(tracks[0].recognition_state == "face_only", "face-only boxes should be labeled for diagnostics");
        check(tracks[0].face.has_value(), "face-only boxes should retain face metadata");
    }

    void test_unassigned_non_webcam_faces_are_ignored() {
        auto cfg = test_face_detector_config();
        FakeFaceDetector detector;
        detector.responses.push_back({face(140.0f, 70.0f, 48.0f, 48.0f)});

        auto state = std::make_shared<veilsight::FaceStateStore>();
        veilsight::HybridFacePolicy policy(cfg, state);
        auto f = frame(32);
        f.source_type = "rtsp";
        std::vector<veilsight::Box> tracks;

        policy.annotate(f, tracks, detector);

        check(tracks.empty(), "unassigned non-webcam faces should not be emitted as face-only boxes");
    }

    void test_face_detector_runs_full_frame_each_frame() {
        auto cfg = test_face_detector_config();
        FakeFaceDetector detector;
        auto state = std::make_shared<veilsight::FaceStateStore>();
        veilsight::HybridFacePolicy policy(cfg, state);

        auto f = frame(1);
        std::vector<veilsight::Box> tracks = {
            person(20.0f, 20.0f, 80.0f, 180.0f, 1),
            person(180.0f, 20.0f, 80.0f, 180.0f, 2),
            person(340.0f, 20.0f, 80.0f, 180.0f, 3),
        };

        policy.annotate(f, tracks, detector);

        auto f2 = frame(2);
        policy.annotate(f2, tracks, detector);

        check(detector.runs.size() == 2, "face detector should run once for each frame");
        check(detector.runs.empty() || detector.runs[0].input_w == 640,
              "face detector should use full-frame detector input size");
        check(tracks[0].privacy_action == "anonymize", "no-face fallback should keep whole person anonymized");
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
    test_identity_transform_maps_geometry_unchanged();
    test_translation_transform_maps_face_to_frame();
    test_full_frame_assignment_chooses_correct_track();
    test_unassigned_webcam_faces_become_face_only_boxes();
    test_unassigned_non_webcam_faces_are_ignored();
    test_face_detector_runs_full_frame_each_frame();
    test_noop_recognizer_passes_tracks_through();

    if (g_failures != 0) {
        std::cerr << "[FAIL] total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "[OK] all face tests passed\n";
    return 0;
}
