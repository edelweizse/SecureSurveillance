#include <anonymization/anonymizer.hpp>
#include <common/config.hpp>
#include <face_detector/face_detector.hpp>
#include <face_detector/face_policy.hpp>
#include <identity/identity_decider.hpp>
#include <person_detector/person_detector.hpp>
#include <pipeline/types.hpp>
#include <recognizer/recognizer.hpp>
#include <tracking/tracker.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace veilsight;

namespace {
    struct Args {
        std::string config_path;
        std::string video_path;
        std::string sequence_id;
        std::string dataset = "ChokePoint";
        std::string system_id = "veilsight";
        std::string output_dir;
        std::string split_mode = "protected";
        std::string gallery_db;
        double deadline_ms = 40.0;
        double fps = 0.0;
    };

    struct Stopwatch {
        using Clock = std::chrono::steady_clock;
        Clock::time_point t0 = Clock::now();

        double lap_ms() {
            const auto now = Clock::now();
            const double ms = std::chrono::duration<double, std::milli>(now - t0).count();
            t0 = now;
            return ms;
        }
    };

    std::string csv_escape(const std::string& value) {
        if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
        std::string out = "\"";
        for (char ch : value) {
            if (ch == '"') out += "\"\"";
            else out += ch;
        }
        out += '"';
        return out;
    }

    template <typename T>
    std::string to_cell(const T& value) {
        std::ostringstream ss;
        ss << value;
        return ss.str();
    }

    template <>
    std::string to_cell<std::string>(const std::string& value) {
        return csv_escape(value);
    }

    void write_row(std::ofstream& out, const std::vector<std::string>& cells) {
        for (size_t i = 0; i < cells.size(); ++i) {
            if (i) out << ',';
            out << csv_escape(cells[i]);
        }
        out << '\n';
    }

    std::string fmt_float(double value) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3) << value;
        return ss.str();
    }

    void usage(const char* prog) {
        std::cerr << "Usage: " << prog << " --config <yaml> --video <mp4> --sequence-id <id> "
                  << "--output-dir <dir> [--dataset ChokePoint] [--system-id veilsight] "
                  << "[--split-mode gallery|protected] [--gallery-db path] [--deadline-ms n] [--fps n]\n";
    }

    Args parse_args(int argc, char** argv) {
        Args args;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const char* name) -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
                return argv[++i];
            };
            if (arg == "--config") args.config_path = require_value("--config");
            else if (arg == "--video") args.video_path = require_value("--video");
            else if (arg == "--sequence-id") args.sequence_id = require_value("--sequence-id");
            else if (arg == "--dataset") args.dataset = require_value("--dataset");
            else if (arg == "--system-id") args.system_id = require_value("--system-id");
            else if (arg == "--output-dir") args.output_dir = require_value("--output-dir");
            else if (arg == "--split-mode") args.split_mode = require_value("--split-mode");
            else if (arg == "--gallery-db") args.gallery_db = require_value("--gallery-db");
            else if (arg == "--deadline-ms") args.deadline_ms = std::stod(require_value("--deadline-ms"));
            else if (arg == "--fps") args.fps = std::stod(require_value("--fps"));
            else if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                std::exit(0);
            } else {
                throw std::runtime_error("unknown option: " + arg);
            }
        }
        if (args.config_path.empty() || args.video_path.empty() || args.sequence_id.empty() || args.output_dir.empty()) {
            throw std::runtime_error("config, video, sequence-id, and output-dir are required");
        }
        if (args.split_mode != "gallery" && args.split_mode != "protected") {
            throw std::runtime_error("split-mode must be gallery or protected");
        }
        if (args.fps < 0.0) {
            throw std::runtime_error("fps must be >= 0");
        }
        return args;
    }

    std::string frame_name(int64_t frame_id) {
        std::ostringstream ss;
        ss << std::setw(8) << std::setfill('0') << frame_id << ".jpg";
        return ss.str();
    }

    std::string region_id(int64_t frame_id, int index) {
        return "r_" + std::to_string(frame_id) + "_" + std::to_string(index);
    }

    bool rect_contains_center(const RectF& face, const Box& body) {
        const float cx = face.x + face.w * 0.5f;
        const float cy = face.y + face.h * 0.5f;
        return cx >= body.x && cx <= body.x + body.w && cy >= body.y && cy <= body.y + body.h;
    }
} // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        usage(argv[0]);
        return 1;
    }

    AppConfig cfg;
    try {
        cfg = load_config_yaml(args.config_path);
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    cfg.modules.face_detector.association_mode = "independent";
    cfg.modules.recognizer.gallery_path = args.gallery_db;

    std::unique_ptr<IPersonDetector> person_detector;
    std::unique_ptr<ITracker> tracker;
    std::unique_ptr<IFaceDetector> face_detector;
    std::unique_ptr<IRecognizer> recognizer;
    std::unique_ptr<IIdentityDecider> identity;
    try {
        person_detector = create_person_detector(cfg.modules.person_detector);
        tracker = create_tracker(cfg.modules.tracker);
        face_detector = create_face_detector(cfg.modules.face_detector);
        recognizer = create_recognizer(cfg.modules.recognizer);
        identity = create_identity_decider(cfg.modules.identity);
    } catch (const std::exception& e) {
        std::cerr << "Model init error: " << e.what() << "\n";
        return 1;
    }

    AnonymizerConfig anon_cfg;
    anon_cfg.method = cfg.runtime.anonymizer.method;
    anon_cfg.pixelation_divisor = cfg.runtime.anonymizer.pixelation_divisor;
    anon_cfg.blur_kernel = cfg.runtime.anonymizer.blur_kernel;
    anon_cfg.face_only_when_available = cfg.runtime.anonymizer.face_only_when_available;
    Anonymizer anonymizer(anon_cfg);

    cv::VideoCapture cap(args.video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << args.video_path << "\n";
        return 1;
    }
    if (args.fps <= 0.0) {
        args.fps = cap.get(cv::CAP_PROP_FPS);
    }
    if (args.fps <= 0.0) {
        args.fps = 25.0;
    }

    const fs::path out_dir(args.output_dir);
    const fs::path frame_dir = out_dir / "output_frames";
    const fs::path mask_dir = out_dir / "masks";
    fs::create_directories(frame_dir);
    fs::create_directories(mask_dir);

    std::ofstream face_log(out_dir / "face_log.csv");
    std::ofstream anon_log(out_dir / "anonymization_log.csv");
    std::ofstream runtime_log(out_dir / "frame_runtime_log.csv");
    std::ofstream body_log(out_dir / "body_log.csv");
    std::ofstream link_log(out_dir / "face_body_link_log.csv");
    if (!face_log || !anon_log || !runtime_log || !body_log || !link_log) {
        std::cerr << "Cannot open output CSV files in " << out_dir << "\n";
        return 1;
    }

    write_row(face_log, {"system_id", "dataset", "sequence_id", "frame_id", "face_track_id", "face_det_id",
                         "face_bbox_x", "face_bbox_y", "face_bbox_w", "face_bbox_h", "face_confidence",
                         "face_size_px", "predicted_identity", "identity_confidence", "recognition_state",
                         "privacy_action", "anonymization_region_id", "output_frame_path"});
    write_row(anon_log, {"system_id", "dataset", "sequence_id", "frame_id", "region_id", "target_type",
                         "source_track_id", "source_track_type", "roi_x", "roi_y", "roi_w", "roi_h",
                         "method", "mask_path"});
    write_row(runtime_log, {"system_id", "dataset", "sequence_id", "frame_id", "input_frame_path",
                            "output_frame_path", "input_frame_received", "output_frame_emitted", "dropped_frame",
                            "latency_ms", "face_detector_ms", "face_tracker_ms", "body_detector_ms",
                            "body_tracker_ms", "recognizer_ms", "anonymizer_ms", "encoder_ms", "deadline_ms",
                            "deadline_missed", "error_message"});
    write_row(body_log, {"system_id", "dataset", "sequence_id", "frame_id", "body_track_id", "body_det_id",
                         "body_bbox_x", "body_bbox_y", "body_bbox_w", "body_bbox_h", "body_confidence",
                         "linked_face_track_id", "body_privacy_action", "anonymization_region_id",
                         "output_frame_path"});
    write_row(link_log, {"system_id", "dataset", "sequence_id", "frame_id", "face_track_id", "body_track_id",
                         "link_method", "face_inside_body", "link_score"});

    std::ofstream config_json(out_dir / "config.json");
    config_json << "{\n"
                << "  \"system_id\": \"" << args.system_id << "\",\n"
                << "  \"dataset\": \"" << args.dataset << "\",\n"
                << "  \"sequence_id\": \"" << args.sequence_id << "\",\n"
                << "  \"split_mode\": \"" << args.split_mode << "\",\n"
                << "  \"base_config\": \"" << args.config_path << "\",\n"
                << "  \"video_path\": \"" << args.video_path << "\",\n"
                << "  \"fps\": " << fmt_float(args.fps) << ",\n"
                << "  \"gallery_db\": \"" << args.gallery_db << "\"\n"
                << "}\n";

    HybridFacePolicy face_policy(cfg.modules.face_detector);

    int64_t frame_id = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        const auto frame_start = Stopwatch::Clock::now();
        Stopwatch stage_timer;
        const fs::path output_path = frame_dir / frame_name(frame_id);
        std::string error_message;

        FramePtr ctx = std::make_shared<FrameCtx>();
        ctx->stream_id = args.sequence_id;
        ctx->source_type = "file";
        ctx->frame_id = frame_id;
        ctx->pts_ns = static_cast<int64_t>((static_cast<double>(frame_id) * 1000000000.0) / args.fps);
        ctx->inf_w = frame.cols;
        ctx->inf_h = frame.rows;
        ctx->ui_w = frame.cols;
        ctx->ui_h = frame.rows;
        ctx->inf = frame.clone();
        ctx->ui = frame.clone();

        double body_detector_ms = 0.0;
        double body_tracker_ms = 0.0;
        double face_detector_ms = 0.0;
        double face_tracker_ms = 0.0;
        double recognizer_ms = 0.0;
        double anonymizer_ms = 0.0;
        double encoder_ms = 0.0;

        std::vector<Box> tracks;
        try {
            const auto detections = person_detector->detect(ctx->inf);
            body_detector_ms = stage_timer.lap_ms();
            ctx->person_detection_count = detections.size();

            TrackerFrameInfo tracker_frame;
            tracker_frame.stream_id = args.sequence_id;
            tracker_frame.frame_id = frame_id;
            tracker_frame.width = frame.cols;
            tracker_frame.height = frame.rows;
            tracks = tracker->update(tracker_frame, detections);
            body_tracker_ms = stage_timer.lap_ms();

            face_policy.annotate(*ctx, tracks, *face_detector);
            face_detector_ms = stage_timer.lap_ms();
            face_tracker_ms = 0.0;
            ctx->face_detection_count = 0;
            for (const auto& track : tracks) {
                if (track.face) ++ctx->face_detection_count;
            }

            RecognitionTask recognition_task;
            recognition_task.stream_id = args.sequence_id;
            recognition_task.frame_id = frame_id;
            recognition_task.frame = ctx;
            recognition_task.tracks = tracks;
            RecognitionResult recognition_result = recognizer->recognize(recognition_task);
            recognizer_ms = stage_timer.lap_ms();

            IdentityTask identity_task;
            identity_task.stream_id = args.sequence_id;
            identity_task.frame_id = frame_id;
            identity_task.frame = ctx;
            identity_task.tracks = recognition_result.tracks;
            IdentityResult identity_result = identity->decide(identity_task);
            tracks = identity_result.tracks;
            ctx->tracked_boxes = tracks;

            const auto regions = anonymizer.planned_regions(ctx->ui, tracks, 1.0f, 1.0f, 0.0f, 0.0f);
            std::map<int, std::string> region_by_track;
            for (size_t i = 0; i < regions.size(); ++i) {
                const std::string rid = region_id(frame_id, static_cast<int>(i));
                region_by_track[regions[i].source_track_id] = rid;
                const fs::path mask_path = mask_dir / (rid + ".png");
                cv::Mat mask = cv::Mat::zeros(ctx->ui.rows, ctx->ui.cols, CV_8UC1);
                cv::rectangle(mask, regions[i].roi, cv::Scalar(255), cv::FILLED);
                cv::imwrite(mask_path.string(), mask);
                write_row(anon_log, {args.system_id, args.dataset, args.sequence_id, std::to_string(frame_id), rid,
                                     regions[i].target_type, std::to_string(regions[i].source_track_id),
                                     regions[i].source_track_type, std::to_string(regions[i].roi.x),
                                     std::to_string(regions[i].roi.y), std::to_string(regions[i].roi.width),
                                     std::to_string(regions[i].roi.height), regions[i].method, mask_path.string()});
            }
            anonymizer.apply(ctx->ui, tracks, 1.0f, 1.0f, 0.0f, 0.0f);
            anonymizer_ms = stage_timer.lap_ms();

            int face_det_id = 0;
            for (const auto& track : tracks) {
                const std::string rid = region_by_track.count(track.id) ? region_by_track[track.id] : "";
                if (track.face) {
                    const auto& face = *track.face;
                    const double face_size = std::max(face.bbox.w, face.bbox.h);
                    write_row(face_log, {args.system_id, args.dataset, args.sequence_id, std::to_string(frame_id),
                                         std::to_string(track.id), std::to_string(face_det_id++),
                                         fmt_float(face.bbox.x), fmt_float(face.bbox.y), fmt_float(face.bbox.w),
                                         fmt_float(face.bbox.h), fmt_float(face.score), fmt_float(face_size),
                                         track.identity_key, fmt_float(track.identity_confidence),
                                         track.recognition_state, track.privacy_action, rid, output_path.string()});
                }
                if (track.id >= 0) {
                    const std::string linked_face = track.face ? std::to_string(track.id) : "";
                    write_row(body_log, {args.system_id, args.dataset, args.sequence_id, std::to_string(frame_id),
                                         std::to_string(track.id), std::to_string(track.id), fmt_float(track.x),
                                         fmt_float(track.y), fmt_float(track.w), fmt_float(track.h), fmt_float(track.score),
                                         linked_face, track.privacy_action, rid, output_path.string()});
                    if (track.face) {
                        const bool inside = rect_contains_center(track.face->bbox, track);
                        write_row(link_log, {args.system_id, args.dataset, args.sequence_id, std::to_string(frame_id),
                                             std::to_string(track.id), std::to_string(track.id), "center_inside_body",
                                             inside ? "1" : "0", inside ? "1.000" : "0.000"});
                    }
                }
            }

            cv::imwrite(output_path.string(), ctx->ui);
            encoder_ms = stage_timer.lap_ms();
        } catch (const std::exception& e) {
            error_message = e.what();
        }

        const double latency_ms = std::chrono::duration<double, std::milli>(
                                      Stopwatch::Clock::now() - frame_start)
                                      .count();
        write_row(runtime_log, {args.system_id, args.dataset, args.sequence_id, std::to_string(frame_id),
                                args.video_path + ":" + std::to_string(frame_id), output_path.string(), "1",
                                error_message.empty() ? "1" : "0", "0", fmt_float(latency_ms),
                                fmt_float(face_detector_ms), fmt_float(face_tracker_ms), fmt_float(body_detector_ms),
                                fmt_float(body_tracker_ms), fmt_float(recognizer_ms), fmt_float(anonymizer_ms),
                                fmt_float(encoder_ms), fmt_float(args.deadline_ms),
                                latency_ms > args.deadline_ms ? "1" : "0", error_message});
        if (!error_message.empty()) {
            std::cerr << "Frame " << frame_id << " error: " << error_message << "\n";
        }
        ++frame_id;
    }

    std::cout << "Processed " << frame_id << " frames into " << out_dir << "\n";
    return 0;
}
