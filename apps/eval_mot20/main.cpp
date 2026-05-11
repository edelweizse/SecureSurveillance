#include <common/config.hpp>
#include <person_detector/person_detector.hpp>
#include <tracking/tracker.hpp>
#include <pipeline/types.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using namespace veilsight;

struct SeqInfo {
    std::string name;
    int seq_length = 0;
    int im_width = 0;
    int im_height = 0;
    std::string im_ext = ".jpg";
};

static std::unordered_map<std::string, std::string> parse_ini(const fs::path& path) {
    std::unordered_map<std::string, std::string> kv;
    std::ifstream in(path);
    if (!in) return kv;
    std::string line;
    while (std::getline(in, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string key = line.substr(0, pos);
        std::string val = line.substr(pos + 1);
        // trim whitespace
        auto not_space = [](int ch) { return !std::isspace(ch); };
        key.erase(key.begin(), std::find_if(key.begin(), key.end(), not_space));
        key.erase(std::find_if(key.rbegin(), key.rend(), not_space).base(), key.end());
        val.erase(val.begin(), std::find_if(val.begin(), val.end(), not_space));
        val.erase(std::find_if(val.rbegin(), val.rend(), not_space).base(), val.end());
        kv[key] = val;
    }
    return kv;
}

static std::vector<std::string> split_comma(const std::string& s) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        auto not_space = [](int ch) { return !std::isspace(ch); };
        part.erase(part.begin(), std::find_if(part.begin(), part.end(), not_space));
        part.erase(std::find_if(part.rbegin(), part.rend(), not_space).base(), part.end());
        if (!part.empty()) parts.push_back(part);
    }
    return parts;
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <config.yaml> [options]\n"
              << "Options:\n"
              << "  --split <train|test>        Dataset split (default: train)\n"
              << "  --sequences <seq1,seq2,...> Comma-separated list or \"all\" (default: all)\n"
              << "  --output <dir>              Tracker output directory (default: results)\n"
              << "  --tracker-name <name>       Tracker folder name (default: veilsight_tracker)\n"
              << "  --detector-thresh <float>   Override detector score threshold\n"
              << "  --detections-only           Write raw detections with unique fake IDs, no tracking\n"
              << "  --help                      Show this message\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string cfg_path;
    std::string split = "train";
    std::string sequences_arg = "all";
    std::string output_dir = "results";
    std::string tracker_name = "veilsight_tracker";
    float detector_thresh_override = -1.0f;
    bool detections_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--split" && i + 1 < argc) {
            split = argv[++i];
        } else if (arg == "--sequences" && i + 1 < argc) {
            sequences_arg = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--tracker-name" && i + 1 < argc) {
            tracker_name = argv[++i];
        } else if (arg == "--detector-thresh" && i + 1 < argc) {
            detector_thresh_override = std::stof(argv[++i]);
        } else if (arg == "--detections-only") {
            detections_only = true;
        } else if (arg.empty() || arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        } else {
            cfg_path = arg;
        }
    }

    if (cfg_path.empty()) {
        std::cerr << "Error: config path required\n";
        print_usage(argv[0]);
        return 1;
    }

    // Load config
    AppConfig cfg;
    try {
        cfg = load_config_yaml(cfg_path);
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    // Force yolox nano as detector per user request
    cfg.modules.person_detector.type = "yolox";
    cfg.modules.person_detector.yolox.variant = "nano";
    if (detector_thresh_override >= 0.0f) {
        cfg.modules.person_detector.yolox.score_threshold = detector_thresh_override;
    }

    // Create detector and tracker
    std::unique_ptr<IPersonDetector> detector;
    std::unique_ptr<ITracker> tracker;
    try {
        detector = create_person_detector(cfg.modules.person_detector);
        if (!detections_only) {
            tracker = create_tracker(cfg.modules.tracker);
        }
    } catch (const std::exception& e) {
        std::cerr << "Model init error: " << e.what() << "\n";
        return 1;
    }

    // Resolve MOT20 base path relative to repo root
    fs::path mot20_base = fs::path("assets") / "MOT20";
    if (!fs::exists(mot20_base)) {
        // try one level up if running from build dir
        mot20_base = fs::path("..") / ".." / ".." / "assets" / "MOT20";
    }
    if (!fs::exists(mot20_base)) {
        std::cerr << "MOT20 dataset not found at " << mot20_base << "\n";
        return 1;
    }

    fs::path split_dir = mot20_base / split;
    if (!fs::exists(split_dir)) {
        std::cerr << "Split directory not found: " << split_dir << "\n";
        return 1;
    }

    // Enumerate sequences
    std::vector<SeqInfo> sequences;
    for (const auto& entry : fs::directory_iterator(split_dir)) {
        if (!entry.is_directory()) continue;
        std::string seq_name = entry.path().filename().string();
        fs::path seqinfo_path = entry.path() / "seqinfo.ini";
        if (!fs::exists(seqinfo_path)) continue;

        auto kv = parse_ini(seqinfo_path);
        SeqInfo info;
        info.name = seq_name;
        info.seq_length = std::stoi(kv.count("seqLength") ? kv.at("seqLength") : "0");
        info.im_width = std::stoi(kv.count("imWidth") ? kv.at("imWidth") : "0");
        info.im_height = std::stoi(kv.count("imHeight") ? kv.at("imHeight") : "0");
        if (kv.count("imExt")) info.im_ext = kv.at("imExt");
        if (info.seq_length > 0) {
            sequences.push_back(info);
        }
    }
    std::sort(sequences.begin(), sequences.end(),
              [](const SeqInfo& a, const SeqInfo& b) { return a.name < b.name; });

    // Filter sequences if requested
    std::vector<SeqInfo> selected;
    if (sequences_arg != "all") {
        auto wanted = split_comma(sequences_arg);
        for (const auto& w : wanted) {
            auto it = std::find_if(sequences.begin(), sequences.end(),
                                   [&w](const SeqInfo& s) { return s.name == w; });
            if (it != sequences.end()) {
                selected.push_back(*it);
            } else {
                std::cerr << "Warning: sequence '" << w << "' not found in " << split_dir << "\n";
            }
        }
    } else {
        selected = sequences;
    }

    if (selected.empty()) {
        std::cerr << "No sequences selected.\n";
        return 1;
    }

    // Prepare output directory
    fs::path tracker_out = fs::path(output_dir) / ("MOT20-" + split) / tracker_name / "data";
    fs::create_directories(tracker_out);

    std::cout << "Evaluating " << selected.size() << " sequence(s) from MOT20-" << split << "\n";
    std::cout << "Detector: yolox_nano (thresh=" << cfg.modules.person_detector.yolox.score_threshold << ")\n";
    if (detections_only) {
        std::cout << "Mode:     detections-only (no tracking)\n";
    } else {
        std::cout << "Tracker:  " << cfg.modules.tracker.type << "\n";
    }
    std::cout << "Output:   " << tracker_out << "\n\n";

    bool any_error = false;
    for (const auto& seq : selected) {
        fs::path img_dir = split_dir / seq.name / "img1";
        fs::path out_file = tracker_out / (seq.name + ".txt");
        std::ofstream out(out_file);
        if (!out) {
            std::cerr << "[ERROR] Cannot write " << out_file << "\n";
            any_error = true;
            continue;
        }

        std::cout << "Processing " << seq.name << " (" << seq.seq_length << " frames)..." << std::flush;
        int written = 0;
        int fake_id = 1;

        for (int t = 1; t <= seq.seq_length; ++t) {
            std::ostringstream img_name;
            img_name << std::setw(6) << std::setfill('0') << t << seq.im_ext;
            fs::path img_path = img_dir / img_name.str();

            cv::Mat frame = cv::imread(img_path.string(), cv::IMREAD_COLOR);
            if (frame.empty()) {
                std::cerr << "\n[WARNING] Missing frame: " << img_path << "\n";
                continue;
            }
            if (frame.cols != seq.im_width || frame.rows != seq.im_height) {
                // warn once per sequence
                if (t == 1) {
                    std::cerr << "\n[WARNING] " << seq.name << " image size (" << frame.cols << "x" << frame.rows
                              << ") differs from seqinfo.ini (" << seq.im_width << "x" << seq.im_height << ")\n";
                }
            }

            auto detections = detector->detect(frame);

            if (detections_only) {
                for (const auto& box : detections) {
                    out << t << ","
                        << fake_id++ << ","
                        << box.x << ","
                        << box.y << ","
                        << box.w << ","
                        << box.h << ","
                        << box.score << ",-1,-1,-1\n";
                    ++written;
                }
            } else {
                TrackerFrameInfo frame_info;
                frame_info.stream_id = seq.name;
                frame_info.frame_id = t;
                frame_info.width = frame.cols;
                frame_info.height = frame.rows;

                auto tracks = tracker->update(frame_info, detections);

                for (const auto& box : tracks) {
                    if (box.id < 1) continue; // skip unconfirmed / invalid
                    out << t << ","
                        << box.id << ","
                        << box.x << ","
                        << box.y << ","
                        << box.w << ","
                        << box.h << ","
                        << box.score << ",-1,-1,-1\n";
                    ++written;
                }
            }
        }

        out.close();
        std::cout << " done (" << written << " detections)\n";
    }

    std::cout << "\nResults written to: " << tracker_out << "\n";
    return any_error ? 1 : 0;
}
