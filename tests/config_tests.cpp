#include <common/config.hpp>
#include <common/replicate.hpp>
#include <streaming/webrtc_publisher.hpp>

#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    int g_failures = 0;

    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++g_failures;
            std::cerr << "[FAIL] " << message << "\n";
        }
    }

    std::string write_yaml_file(const std::string& prefix, const std::string& body) {
        namespace fs = std::filesystem;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const fs::path path = fs::temp_directory_path() /
                              (prefix + "_" + std::to_string(stamp) + ".yaml");

        std::ofstream out(path);
        if (!out.is_open()) {
            throw std::runtime_error("failed to open temp config file: " + path.string());
        }
        out << body;
        out.close();
        return path.string();
    }

    std::filesystem::path find_repo_file(const std::filesystem::path& relative) {
        namespace fs = std::filesystem;
        fs::path dir = fs::current_path();
        for (int i = 0; i < 8; ++i) {
            const fs::path candidate = dir / relative;
            if (fs::exists(candidate)) return candidate;
            if (!dir.has_parent_path()) break;
            dir = dir.parent_path();
        }

        const fs::path source_relative = fs::path(__FILE__).parent_path().parent_path() / relative;
        if (fs::exists(source_relative)) return source_relative;
        return relative;
    }

    bool load_throws(const std::string& yaml) {
        const std::string path = write_yaml_file("veilsight_cfg", yaml);
        try {
            (void)veilsight::load_config_yaml(path);
            std::filesystem::remove(path);
            return false;
        } catch (...) {
            std::filesystem::remove(path);
            return true;
        }
    }

    std::string minimal_config_yaml(const std::string& extra) {
        return
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n" +
            extra +
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    outputs:\n"
            "      fps: 12\n"
            "      profiles:\n"
            "        inference:\n"
            "          width: 640\n"
            "          height: 640\n"
            "        ui:\n"
            "          width: 1280\n"
            "          height: 720\n";
    }

    void test_expand_replicas_fills_missing_ids() {
        veilsight::IngestConfig in_cfg;
        in_cfg.id = "cam0";
        in_cfg.type = "webcam";
        in_cfg.replicate.count = 3;
        in_cfg.replicate.ids = {"custom_0"};

        const std::vector<veilsight::IngestConfig> input = {in_cfg};
        const auto expanded = veilsight::expand_replicas(input);

        check(expanded.size() == 3, "expand_replicas should output replicate.count entries");
        check(expanded[0].id == "custom_0", "expand_replicas should preserve provided ids");
        check(expanded[1].id == "cam0_1", "expand_replicas should synthesize missing id #1");
        check(expanded[2].id == "cam0_2", "expand_replicas should synthesize missing id #2");
    }

    void test_config_rejects_legacy_output() {
        const std::string yaml =
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    output:\n"
            "      width: 1280\n"
            "      height: 720\n";

        check(load_throws(yaml), "load_config_yaml should reject legacy stream.output schema");
    }

    void test_config_requires_global_outputs_fps() {
        const std::string yaml =
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    outputs:\n"
            "      profiles:\n"
            "        inference:\n"
            "          width: 640\n"
            "          height: 640\n"
            "        ui:\n"
            "          width: 1280\n"
            "          height: 720\n";

        check(load_throws(yaml), "load_config_yaml should require outputs.fps > 0");
    }

    void test_global_outputs_fps_overrides_profile_fps() {
        const std::string yaml =
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    outputs:\n"
            "      fps: 12\n"
            "      profiles:\n"
            "        inference:\n"
            "          width: 640\n"
            "          height: 640\n"
            "          fps: 5\n"
            "        ui:\n"
            "          width: 1280\n"
            "          height: 720\n"
            "          fps: 30\n";

        const std::string path = write_yaml_file("veilsight_cfg_ok", yaml);
        const auto cfg = veilsight::load_config_yaml(path);
        std::filesystem::remove(path);

        check(cfg.streams.size() == 1, "valid config should load exactly one stream");
        const auto& profiles = cfg.streams[0].outputs.profiles;
        check(cfg.streams[0].outputs.fps == 12, "outputs.fps should be stored");
        check(profiles.at("inference").fps == 12, "inference fps should be synchronized to outputs.fps");
        check(profiles.at("ui").fps == 12, "ui fps should be synchronized to outputs.fps");
    }

    std::string base_streaming_yaml(const std::string& streaming) {
        return
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "streaming:\n" + streaming +
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    outputs:\n"
            "      fps: 12\n"
            "      profiles:\n"
            "        inference:\n"
            "          width: 640\n"
            "          height: 640\n"
            "        ui:\n"
            "          width: 1280\n"
            "          height: 720\n";
    }

    void test_streaming_webrtc_defaults_and_parsing() {
        const std::string yaml = base_streaming_yaml(
            "  primary: \"webrtc\"\n"
            "  fallback: \"mjpeg\"\n"
            "  codec: \"h264\"\n"
            "  encoder: \"auto\"\n"
            "  bitrate_kbps: 6000\n"
            "  keyframe_interval_frames: 30\n"
            "  webrtc:\n"
            "    enabled: true\n"
            "    max_peers_per_stream: 3\n"
            "    ice_gathering_timeout_ms: 1500\n"
            "    session_idle_timeout_s: 45\n"
            "    stun_servers:\n"
            "      - \"stun:stun.l.google.com:19302\"\n"
            "    cors_allowed_origins:\n"
            "      - \"http://localhost:8000\"\n");

        const std::string path = write_yaml_file("veilsight_streaming_ok", yaml);
        const auto cfg = veilsight::load_config_yaml(path);
        std::filesystem::remove(path);

        check(cfg.streaming.primary == "webrtc", "streaming.primary should parse");
        check(cfg.streaming.webrtc.max_peers_per_stream == 3, "webrtc peer limit should parse");
        check(cfg.streaming.webrtc.ice_gathering_timeout_ms == 1500, "ICE timeout should parse");
        check(cfg.streaming.webrtc.stun_servers.size() == 1, "STUN servers should parse");
    }

    void test_streaming_webrtc_default_cors_allows_vite_loopback() {
        const std::string yaml = base_streaming_yaml(
            "  primary: \"webrtc\"\n"
            "  fallback: \"mjpeg\"\n"
            "  codec: \"h264\"\n"
            "  encoder: \"auto\"\n"
            "  bitrate_kbps: 6000\n"
            "  keyframe_interval_frames: 30\n");

        const std::string path = write_yaml_file("veilsight_streaming_cors_defaults", yaml);
        const auto cfg = veilsight::load_config_yaml(path);
        std::filesystem::remove(path);

        const auto& origins = cfg.streaming.webrtc.cors_allowed_origins;
        check(std::find(origins.begin(), origins.end(), "http://localhost:5173") != origins.end(),
              "default CORS origins should allow localhost Vite dev server");
        check(std::find(origins.begin(), origins.end(), "http://127.0.0.1:5173") != origins.end(),
              "default CORS origins should allow 127.0.0.1 Vite dev server");
    }

    void test_streaming_validation_rejects_invalid_values() {
        check(load_throws(base_streaming_yaml("  bitrate_kbps: 0\n")),
              "streaming.bitrate_kbps must reject non-positive values");
        check(load_throws(base_streaming_yaml("  codec: \"vp8\"\n")),
              "streaming.codec must reject non-h264 values");
        check(load_throws(base_streaming_yaml("  webrtc:\n    max_peers_per_stream: 0\n")),
              "streaming.webrtc.max_peers_per_stream must reject zero");
        check(load_throws(base_streaming_yaml("  webrtc:\n    ice_gathering_timeout_ms: 50\n")),
              "streaming.webrtc.ice_gathering_timeout_ms must reject low timeout");
    }

    void test_h264_encoder_selector() {
        veilsight::StreamingConfig cfg;
        cfg.encoder = "x264enc";
        veilsight::H264EncoderSelector exact([](const std::string& name) {
            return name == "webrtcbin" || name == "x264enc";
        });
        auto selected = exact.select(cfg);
        check(selected.available && selected.encoder == "x264enc",
              "exact configured encoder should be selected when available");

        cfg.encoder = "auto";
        veilsight::H264EncoderSelector automatic([](const std::string& name) {
            return name == "webrtcbin" || name == "openh264enc";
        });
        selected = automatic.select(cfg);
        check(selected.available && selected.encoder == "openh264enc",
              "auto selector should pick the first available fake encoder");

        cfg.encoder = "missingenc";
        selected = exact.select(cfg);
        check(!selected.available && selected.error.find("missingenc") != std::string::npos,
              "missing exact encoder should return a clear error");
    }

    void test_person_detector_and_scene_grid_config() {
        const std::string yaml =
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "modules:\n"
            "  person_detector:\n"
            "    type: \"yolox\"\n"
            "    workers: 2\n"
            "    yolox:\n"
            "      variant: \"tiny\"\n"
            "      param_path: \"models/people_detectors/yolox_tiny/bytetrack_tiny.ncnn.param\"\n"
            "      bin_path: \"models/people_detectors/yolox_tiny/bytetrack_tiny.ncnn.bin\"\n"
            "      class_id: 3\n"
            "      ncnn_threads: 4\n"
            "  tracker:\n"
            "    type: \"bytetrack\"\n"
            "    bytetrack:\n"
            "      min_box_area: 144\n"
            "      scene_grid:\n"
            "        enabled: true\n"
            "        rows: 5\n"
            "        cols: 7\n"
            "        association_weight: 0.2\n"
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    outputs:\n"
            "      fps: 12\n"
            "      profiles:\n"
            "        inference:\n"
            "          width: 640\n"
            "          height: 640\n"
            "        ui:\n"
            "          width: 1280\n"
            "          height: 720\n";

        const std::string path = write_yaml_file("veilsight_person_cfg_ok", yaml);
        const auto cfg = veilsight::load_config_yaml(path);
        std::filesystem::remove(path);

        check(cfg.modules.person_detector.type == "yolox", "detector.type should parse yolox");
        check(cfg.modules.person_detector.yolox.variant == "tiny", "yolox.variant should parse");
        check(cfg.modules.person_detector.yolox.param_path == "models/people_detectors/yolox_tiny/bytetrack_tiny.ncnn.param",
              "yolox.param_path should parse");
        check(cfg.modules.person_detector.yolox.bin_path == "models/people_detectors/yolox_tiny/bytetrack_tiny.ncnn.bin",
              "yolox.bin_path should parse");
        check(cfg.modules.person_detector.yolox.class_id == 3, "yolox.class_id should parse");
        check(cfg.modules.person_detector.yolox.ncnn_threads == 4, "yolox.ncnn_threads should parse");
        check(cfg.modules.tracker.bytetrack.min_box_area == 144.0f, "bytetrack min_box_area should parse");
        check(cfg.modules.tracker.bytetrack.scene_grid.rows == 5, "scene_grid rows should parse");
        check(cfg.modules.tracker.bytetrack.scene_grid.cols == 7, "scene_grid cols should parse");
        check(cfg.modules.tracker.bytetrack.scene_grid.association_weight == 0.2f,
              "scene_grid association_weight should parse");
    }

    void test_legacy_person_class_id_alias() {
        const std::string yaml =
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "modules:\n"
            "  person_detector:\n"
            "    yolox:\n"
            "      person_class_id: 2\n"
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    outputs:\n"
            "      fps: 12\n"
            "      profiles:\n"
            "        inference:\n"
            "          width: 640\n"
            "          height: 640\n"
            "        ui:\n"
            "          width: 1280\n"
            "          height: 720\n";

        const std::string path = write_yaml_file("veilsight_person_alias_cfg_ok", yaml);
        const auto cfg = veilsight::load_config_yaml(path);
        std::filesystem::remove(path);

        check(cfg.modules.person_detector.yolox.class_id == 2, "person_class_id should remain an alias for class_id");
    }

    void test_face_detector_recognizer_identity_config_parses() {
        const std::string yaml =
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "modules:\n"
            "  person_detector:\n"
            "    type: \"yolox\"\n"
            "    workers: 3\n"
            "    yolox:\n"
            "      ncnn_threads: 2\n"
            "  face_detector:\n"
            "    enabled: true\n"
            "    type: \"scrfd\"\n"
            "    workers: 2\n"
            "    scrfd:\n"
            "      variant: \"500m_landmarks\"\n"
            "      input_w: 640\n"
            "      input_h: 640\n"
            "      score_threshold: 0.45\n"
            "      nms_threshold: 0.30\n"
            "      top_k: 100\n"
            "      ncnn_threads: 1\n"
            "  recognizer:\n"
            "    type: \"noop\"\n"
            "    workers: 9\n"
            "    gallery_path: \"/legacy/gallery\"\n"
            "  identity:\n"
            "    type: \"noop\"\n"
            "    workers: 4\n"
            "    gallery_path: \"/identity/gallery\"\n"
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    outputs:\n"
            "      fps: 12\n"
            "      profiles:\n"
            "        inference:\n"
            "          width: 640\n"
            "          height: 640\n"
            "        ui:\n"
            "          width: 1280\n"
            "          height: 720\n";

        const std::string path = write_yaml_file("veilsight_face_recognizer_cfg_ok", yaml);
        const auto cfg = veilsight::load_config_yaml(path);
        std::filesystem::remove(path);

        check(cfg.modules.person_detector.workers == 3, "person detector worker count should parse independently");
        check(cfg.modules.person_detector.yolox.ncnn_threads == 2,
              "person detector internal threads should parse independently");
        check(cfg.modules.recognizer.type == "noop", "recognizer.type should parse noop");
        check(cfg.modules.recognizer.workers == 9, "recognizer worker count should parse");
        check(cfg.modules.recognizer.gallery_path == "/legacy/gallery",
              "recognizer.gallery_path should parse without mapping into identity");
        check(cfg.modules.face_detector.type == "scrfd", "face detector type should parse");
        check(cfg.modules.face_detector.workers == 2,
              "face detector worker count should parse independently");
        check(cfg.modules.face_detector.scrfd.variant == "500ml",
              "face SCRFD variant should parse");
        check(cfg.modules.face_detector.scrfd.ncnn_threads == 1,
              "face detector internal threads should parse independently");
        check(cfg.modules.face_detector.scrfd.param_path.find("scrfd_500m_l") != std::string::npos,
              "face SCRFD variant should select landmark model paths");
        check(cfg.modules.face_detector.scrfd.top_k == 100, "face SCRFD top_k should parse");
        check(cfg.modules.face_detector.scrfd.input_w == 640,
              "face detector input width should parse");
        check(cfg.modules.identity.type == "noop", "identity.type should parse noop");
        check(cfg.modules.identity.workers == 4, "identity worker count should parse");
        check(cfg.modules.identity.gallery_path == "/identity/gallery",
              "identity.gallery_path should not inherit recognizer.gallery_path");
    }

    void test_mobilefacenet_recognizer_config_parses_and_validates() {
        const std::string yaml = minimal_config_yaml(
            "modules:\n"
            "  recognizer:\n"
            "    type: \"mobilefacenet\"\n"
            "    model_instances: 2\n"
            "    gallery_path: \"/tmp/gallery.sqlite3\"\n"
            "    param_path: \"models/face_embeddings/mobilefacenet/mobilefacenets.param\"\n"
            "    bin_path: \"models/face_embeddings/mobilefacenet/mobilefacenets.bin\"\n"
            "    input_blob: \"data\"\n"
            "    output_blob: \"fc1\"\n"
            "    input_w: 112\n"
            "    input_h: 112\n"
            "    embedding_dim: 128\n"
            "    ncnn_threads: 3\n"
            "    cache_ttl_frames: 123\n"
            "    min_face_score: 0.71\n"
            "    min_face_size_px: 57.0\n"
            "    min_inter_eye_px: 21.0\n"
            "    max_roll_deg: 24.0\n"
            "    max_yaw_offset_ratio: 0.34\n"
            "  identity:\n"
            "    type: \"passthrough\"\n");

        const std::string path = write_yaml_file("veilsight_mobilefacenet_cfg_ok", yaml);
        const auto cfg = veilsight::load_config_yaml(path);
        std::filesystem::remove(path);

        check(cfg.modules.recognizer.type == "mobilefacenet", "recognizer.type should parse mobilefacenet");
        check(cfg.modules.recognizer.workers == 2, "mobilefacenet model_instances should parse");
        check(cfg.modules.recognizer.gallery_path == "/tmp/gallery.sqlite3", "mobilefacenet gallery_path should parse");
        check(cfg.modules.recognizer.unknown_threshold == 0.45f,
              "mobilefacenet unknown_threshold should default to 0.45");
        check(cfg.modules.recognizer.input_blob == "data", "mobilefacenet input_blob should parse");
        check(cfg.modules.recognizer.output_blob == "fc1", "mobilefacenet output_blob should parse");
        check(cfg.modules.recognizer.input_w == 112 && cfg.modules.recognizer.input_h == 112,
              "mobilefacenet input size should parse");
        check(cfg.modules.recognizer.embedding_dim == 128, "mobilefacenet embedding_dim should parse");
        check(cfg.modules.recognizer.ncnn_threads == 3, "mobilefacenet ncnn_threads should parse");
        check(cfg.modules.recognizer.cache_ttl_frames == 123, "mobilefacenet cache_ttl_frames should parse");
        check(cfg.modules.recognizer.min_face_score == 0.71f, "mobilefacenet min_face_score should parse");
        check(cfg.modules.recognizer.min_face_size_px == 57.0f, "mobilefacenet min_face_size_px should parse");
        check(cfg.modules.recognizer.min_inter_eye_px == 21.0f, "mobilefacenet min_inter_eye_px should parse");
        check(cfg.modules.recognizer.max_roll_deg == 24.0f, "mobilefacenet max_roll_deg should parse");
        check(cfg.modules.recognizer.max_yaw_offset_ratio == 0.34f,
              "mobilefacenet max_yaw_offset_ratio should parse");
        check(cfg.modules.identity.type == "passthrough", "identity.type should parse passthrough");

        check(load_throws(minimal_config_yaml(
                  "modules:\n"
                  "  recognizer:\n"
                  "    type: \"mobilefacenet\"\n"
                  "    input_w: 111\n")),
              "mobilefacenet should reject non-112 input width");
        check(load_throws(minimal_config_yaml(
                  "modules:\n"
                  "  recognizer:\n"
                  "    type: \"mobilefacenet\"\n"
                  "    embedding_dim: 127\n")),
              "mobilefacenet should reject non-128 embedding_dim");
        check(load_throws(minimal_config_yaml(
                  "modules:\n"
                  "  recognizer:\n"
                  "    type: \"noop\"\n"
                  "    ncnn_threads: 0\n")),
              "recognizer internal thread count should reject zero");
    }

    void test_scrfd_variant_aliases_resolve() {
        struct Case {
            std::string alias;
            std::string canonical;
            std::string path_fragment;
        };
        const std::vector<Case> cases = {
            {"2.5g", "2g", "scrfd_2g"},
            {"25g", "2g", "scrfd_2g"},
            {"2.5g_landmarks", "2gl", "scrfd_2g_l"},
            {"25g_landmarks", "2gl", "scrfd_2g_l"},
            {"500m_landmarks", "500ml", "scrfd_500m_l"},
            {"10g", "10g", "scrfd_10g"},
        };

        for (const auto& c : cases) {
            const std::string yaml =
                "server:\n"
                "  host: \"0.0.0.0\"\n"
                "  port: 8080\n"
                "modules:\n"
                "  person_detector:\n"
                "    scrfd:\n"
                "      variant: \"" + c.alias + "\"\n"
                "streams:\n"
                "  - id: \"file0\"\n"
                "    type: \"file\"\n"
                "    file:\n"
                "      path: \"/tmp/test.mp4\"\n"
                "    outputs:\n"
                "      fps: 12\n"
                "      profiles:\n"
                "        inference:\n"
                "          width: 640\n"
                "          height: 640\n"
                "        ui:\n"
                "          width: 1280\n"
                "          height: 720\n";

            const std::string path = write_yaml_file("veilsight_scrfd_alias_cfg_ok", yaml);
            const auto cfg = veilsight::load_config_yaml(path);
            std::filesystem::remove(path);

            check(cfg.modules.person_detector.scrfd.variant == c.canonical,
                  "SCRFD alias should canonicalize: " + c.alias);
            check(cfg.modules.person_detector.scrfd.param_path.find(c.path_fragment) != std::string::npos,
                  "SCRFD alias should select param path: " + c.alias);
            check(cfg.modules.person_detector.scrfd.bin_path.find(c.path_fragment) != std::string::npos,
                  "SCRFD alias should select bin path: " + c.alias);
        }
    }

    void test_full_reference_config_loads() {
        const auto path = find_repo_file("configs/full_reference.yaml");
        const auto cfg = veilsight::load_config_yaml(path.string());

        check(cfg.streams.size() == 1, "full_reference should configure exactly one active stream");
        check(cfg.streams[0].id == "file0", "full_reference active stream should be file0");
        check(cfg.streams[0].type == "file", "full_reference active stream should be a file stream");
        check(cfg.modules.person_detector.type == "yolox", "modules.person_detector should parse into detector config");
        check(cfg.modules.person_detector.workers == 2, "person_detector.model_instances should parse");
        check(cfg.modules.face_detector.type == "scrfd",
              "modules.face_detector should parse into face detector config");
        check(cfg.modules.face_detector.workers == 2,
              "face_detector.model_instances should parse");
        check(cfg.modules.face_detector.scrfd.input_w == 640,
              "face detector input should parse");
        check(cfg.modules.recognizer.type == "mobilefacenet",
              "modules.recognizer should parse mobilefacenet");
        check(cfg.modules.identity.type == "passthrough",
              "modules.identity should parse passthrough");
        check(cfg.runtime.queues.global.person_detector_in_capacity == 50,
              "runtime global person detector queue capacity should parse");
        check(cfg.runtime.queues.global.recognizer_in_capacity == 50,
              "runtime global recognizer queue capacity should parse");
        check(cfg.runtime.queues.per_stream.recognitions_in_capacity == 20,
              "runtime per-stream recognitions queue capacity should parse");
        check(cfg.runtime.queues.per_stream.encoder_in_capacity == 5,
              "runtime per-stream encoder queue capacity should parse");
        check(cfg.runtime.anonymizer.model_instances == 1,
              "runtime anonymizer model_instances should parse");
        check(cfg.runtime.anonymizer.method == "pixelate",
              "runtime anonymizer method should parse");
    }

    void test_model_instances_aliases_workers() {
        const std::string preferred_yaml = minimal_config_yaml(
            "modules:\n"
            "  person_detector:\n"
            "    type: \"yolox\"\n"
            "    workers: 1\n"
            "    model_instances: 3\n");
        const std::string preferred_path = write_yaml_file("veilsight_model_instances", preferred_yaml);
        const auto preferred_cfg = veilsight::load_config_yaml(preferred_path);
        std::filesystem::remove(preferred_path);
        check(preferred_cfg.modules.person_detector.workers == 3,
              "person_detector.model_instances should win over workers");

        const std::string workers_yaml = minimal_config_yaml(
            "modules:\n"
            "  person_detector:\n"
            "    type: \"yolox\"\n"
            "    workers: 2\n");
        const std::string workers_path = write_yaml_file("veilsight_workers_alias", workers_yaml);
        const auto workers_cfg = veilsight::load_config_yaml(workers_path);
        std::filesystem::remove(workers_path);
        check(workers_cfg.modules.person_detector.workers == 2,
              "person_detector.workers should still parse");
    }

    void test_face_detector_top_level_config_parses() {
        const std::string yaml = minimal_config_yaml(
            "modules:\n"
            "  face_detector:\n"
            "    enabled: true\n"
            "    type: \"scrfd\"\n"
            "    model_instances: 3\n"
            "    scrfd:\n"
            "      variant: \"25g_landmarks\"\n"
            "      input_w: 512\n"
            "      ncnn_threads: 2\n");

        const std::string path = write_yaml_file("veilsight_face_top_level", yaml);
        const auto cfg = veilsight::load_config_yaml(path);
        std::filesystem::remove(path);

        check(cfg.modules.face_detector.type == "scrfd", "top-level face detector type should parse");
        check(cfg.modules.face_detector.workers == 3,
              "top-level face_detector.model_instances should parse");
        check(cfg.modules.face_detector.scrfd.variant == "2gl",
              "top-level face detector SCRFD variant should parse");
        check(cfg.modules.face_detector.scrfd.ncnn_threads == 2,
              "top-level face detector ncnn_threads should parse");
        check(cfg.modules.face_detector.scrfd.input_w == 512,
              "top-level face detector input should parse");
    }

    void test_face_detector_can_be_disabled() {
        const std::string yaml = minimal_config_yaml(
            "modules:\n"
            "  face_detector:\n"
            "    enabled: false\n"
            "    type: \"scrfd\"\n"
            "    model_instances: 2\n");

        const std::string path = write_yaml_file("veilsight_face_disabled", yaml);
        const auto cfg = veilsight::load_config_yaml(path);
        std::filesystem::remove(path);

        check(cfg.modules.face_detector.type == "none",
              "disabled top-level face_detector should set detector type to none");
        check(cfg.modules.recognizer.type == "noop",
              "disabled top-level face_detector should not alter recognizer config");
    }

    void test_legacy_inference_config_paths_are_rejected() {
        check(load_throws(minimal_config_yaml(
                  "modules:\n"
                  "  detector:\n"
                  "    type: \"yolox\"\n")),
              "legacy modules.detector should be unsupported");

        check(load_throws(minimal_config_yaml(
                  "modules:\n"
                  "  recognizer:\n"
                  "    face:\n"
                  "      detector:\n"
                  "        type: \"scrfd\"\n")),
              "legacy modules.recognizer.face should be unsupported");
    }

    void test_runtime_config_parses() {
        const std::string yaml = minimal_config_yaml(
            "runtime:\n"
            "  reorder_window: 7\n"
            "  pending_state_limit: 333\n"
            "  jpeg_quality: 81\n"
            "  queues:\n"
            "    global:\n"
            "      person_detector_in_capacity: 51\n"
            "      face_detector_in_capacity: 52\n"
            "      recognizer_in_capacity: 53\n"
            "      identity_in_capacity: 54\n"
            "      anonymizer_in_capacity: 55\n"
            "    per_stream:\n"
            "      frames_in_capacity: 6\n"
            "      person_detections_in_capacity: 21\n"
            "      faces_in_capacity: 22\n"
            "      recognitions_in_capacity: 23\n"
            "      identities_in_capacity: 24\n"
            "      encoder_in_capacity: 7\n"
            "  anonymizer:\n"
            "    model_instances: 4\n"
            "    method: \"blur\"\n"
            "    pixelation_divisor: 12\n"
            "    blur_kernel: 33\n");

        const std::string path = write_yaml_file("veilsight_runtime", yaml);
        const auto cfg = veilsight::load_config_yaml(path);
        std::filesystem::remove(path);

        check(cfg.runtime.reorder_window == 7, "runtime.reorder_window should parse");
        check(cfg.runtime.pending_state_limit == 333, "runtime.pending_state_limit should parse");
        check(cfg.runtime.jpeg_quality == 81, "runtime.jpeg_quality should parse");
        check(cfg.runtime.queues.global.person_detector_in_capacity == 51,
              "runtime person_detector_in_capacity should parse");
        check(cfg.runtime.queues.global.face_detector_in_capacity == 52,
              "runtime face_detector_in_capacity should parse");
        check(cfg.runtime.queues.global.recognizer_in_capacity == 53,
              "runtime recognizer_in_capacity should parse");
        check(cfg.runtime.queues.global.identity_in_capacity == 54,
              "runtime identity_in_capacity should parse");
        check(cfg.runtime.queues.global.anonymizer_in_capacity == 55,
              "runtime anonymizer_in_capacity should parse");
        check(cfg.runtime.queues.per_stream.frames_in_capacity == 6,
              "runtime frames_in_capacity should parse");
        check(cfg.runtime.queues.per_stream.person_detections_in_capacity == 21,
              "runtime person_detections_in_capacity should parse");
        check(cfg.runtime.queues.per_stream.faces_in_capacity == 22,
              "runtime faces_in_capacity should parse");
        check(cfg.runtime.queues.per_stream.recognitions_in_capacity == 23,
              "runtime recognitions_in_capacity should parse");
        check(cfg.runtime.queues.per_stream.identities_in_capacity == 24,
              "runtime identities_in_capacity should parse");
        check(cfg.runtime.queues.per_stream.encoder_in_capacity == 7,
              "runtime encoder_in_capacity should parse");
        check(cfg.runtime.anonymizer.model_instances == 4,
              "runtime anonymizer.model_instances should parse");
        check(cfg.runtime.anonymizer.method == "blur", "runtime anonymizer.method should parse");
        check(cfg.runtime.anonymizer.pixelation_divisor == 12,
              "runtime anonymizer.pixelation_divisor should parse");
        check(cfg.runtime.anonymizer.blur_kernel == 33,
              "runtime anonymizer.blur_kernel should parse");
    }

    void test_runtime_config_rejects_invalid_values() {
        check(load_throws(minimal_config_yaml(
                  "runtime:\n"
                  "  queues:\n"
                  "    global:\n"
                  "      person_detector_in_capacity: 0\n")),
              "runtime queue capacities must reject zero");
        check(load_throws(minimal_config_yaml(
                  "runtime:\n"
                  "  jpeg_quality: 0\n")),
              "runtime.jpeg_quality must reject zero");
        check(load_throws(minimal_config_yaml(
                  "runtime:\n"
                  "  jpeg_quality: 101\n")),
              "runtime.jpeg_quality must reject values above 100");
        check(load_throws(minimal_config_yaml(
                  "runtime:\n"
                  "  anonymizer:\n"
                  "    method: \"mask\"\n")),
              "runtime.anonymizer.method must reject unknown methods");
        check(load_throws(minimal_config_yaml(
                  "modules:\n"
                  "  person_detector:\n"
                  "    type: \"yolox\"\n"
                  "    yolox:\n"
                  "      ncnn_threads: 0\n")),
              "internal model thread counts must reject zero");
    }
}

int main() {
    test_expand_replicas_fills_missing_ids();
    test_config_rejects_legacy_output();
    test_config_requires_global_outputs_fps();
    test_global_outputs_fps_overrides_profile_fps();
    test_streaming_webrtc_defaults_and_parsing();
    test_streaming_webrtc_default_cors_allows_vite_loopback();
    test_streaming_validation_rejects_invalid_values();
    test_h264_encoder_selector();
    test_person_detector_and_scene_grid_config();
    test_legacy_person_class_id_alias();
    test_face_detector_recognizer_identity_config_parses();
    test_mobilefacenet_recognizer_config_parses_and_validates();
    test_scrfd_variant_aliases_resolve();
    test_full_reference_config_loads();
    test_model_instances_aliases_workers();
    test_face_detector_top_level_config_parses();
    test_face_detector_can_be_disabled();
    test_legacy_inference_config_paths_are_rejected();
    test_runtime_config_parses();
    test_runtime_config_rejects_invalid_values();

    if (g_failures != 0) {
        std::cerr << "[FAIL] total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "[OK] all config tests passed\n";
    return 0;
}
