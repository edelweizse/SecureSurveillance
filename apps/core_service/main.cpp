#include <common/config.hpp>
#include <common/resize.hpp>
#include <encode/mjpeg_server.hpp>
#include <ingest/frame_source_factory.hpp>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/exceptions.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <csignal>

static std::atomic<bool> g_running(true);
static void handle_sigint(int) { g_running = false; }

std::vector<ss::IngestConfig> expand_replicas(const std::vector<ss::IngestConfig>& in) {
    std::vector<ss::IngestConfig> out;
    for (const auto& s : in) {
        int n = std::max(1, s.replicate.count);

        if (n == 1) {
            auto one = s;
            one.replicate.count = 1;
            out.push_back(std::move(one));
            continue;
        }

        std::vector<std::string> ids = s.replicate.ids;
        if (ids.empty()) {
            ids.reserve(n);
            for (int i = 0; i < n; i++) ids.push_back(s.id + "_" + std::to_string(i));
        }

        for (int i = 0; i < n; i++) {
            auto r = s;
            r.id = ids[i];
            r.replicate.count = 1;
            r.replicate.ids.clear();

            // appsink name must be unique
            out.push_back(std::move(r));
        }
    }
    return out;
}

static std::vector<std::pair<std::string, ss::OutputConfig>> output_profiles(const ss::IngestConfig& cfg) {
    std::vector<std::pair<std::string, ss::OutputConfig>> out;

    if (!cfg.outputs.profiles.empty()) {
        out.reserve(cfg.outputs.profiles.size());
        for (const auto& kv : cfg.outputs.profiles) {
            out.push_back({kv.first, kv.second});
        }
        std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.first < b.first; });
        return out;
    }

    out.push_back({"main", cfg.output});
    return out;
}

static void run_src_worker(const ss::IngestConfig& cfg, ss::MJPEGServer& server) {
    std::unique_ptr<ss::IFrameSource> src;
    try {
        src = ss::make_frame_source(cfg);
    } catch (const std::exception& e) {
        std::cerr << "[worker:" << cfg.id << "] Failed to create source: " << e.what() << "\n";
        return;
    }

    if (!src->start()) {
        std::cerr << "[worker:" << cfg.id << "] Failed to start source.\n";
        return;
    }

    const auto profiles = output_profiles(cfg);

    ss::FramePacket fp;
    int empty_count = 0;

    while (g_running) {
        if (!src->read(fp, 5000)) {
            // if (++empty_count % 120 == 0) {
            //     std::cerr << "[worker:" + cfg.id + "] No frames yet..\n";
            // }
            continue;
        }
        empty_count = 0;

        if (fp.bgr.empty()) continue;

        for (const auto& pr : profiles) {
            const std::string& profile = pr.first;
            const ss::OutputConfig oc = pr.second;

            const int interp = ss::interp_from_str(oc.interp);

            cv::Mat out = ss::resize_frame(
                fp.bgr,
                oc.width,
                oc.height,
                oc.keep_aspect,
                interp
            );

            std::vector<int> jpeg_params = { cv::IMWRITE_JPEG_QUALITY, oc.jpg_quality };
            std::vector<uint8_t> jpeg;
            if (!cv::imencode(".jpg", out, jpeg, jpeg_params)) {
                std::cerr << "[worker:" << cfg.id << "] imencode failed for profile = " << profile << "\n";
                continue;
            }

            auto jpeg_ptr = std::make_shared<const std::vector<uint8_t>>(std::move(jpeg));
            const std::string stream_key = cfg.id + "/" + profile;

            std::string stream_meta =
                std::string("{") +
                R"("stream_id":")" + cfg.id + "\"," +
                R"("profile":")" + profile + "\"," +
                R"("frame_id":)" + std::to_string(fp.frame_id) + "," +
                R"("pts_ns":)" + std::to_string(fp.pts_ns) + "," +
                R"("w":)" + std::to_string(out.cols) + "," +
                R"("h":)" + std::to_string(out.rows) +
                std::string("}");

            server.push_jpeg(stream_key, jpeg_ptr);
            server.push_meta(stream_key, std::move(stream_meta));
        }
    }

    src->stop();
    std::cerr << "[worker:" << cfg.id << "] Stopped.\n";
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    std::string cfg_path = "../../../configs/multi.yaml";
    if (argc >= 2) cfg_path = argv[1];
    else std::cerr << "Using default config: " << cfg_path << "\n";

    ss::AppConfig cfg;
    try {
        cfg = ss::load_config_yaml(cfg_path);
    } catch (const YAML::Exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    std::vector<ss::IngestConfig> streams = expand_replicas(cfg.streams);
    if (streams.empty()) {
        std::cerr << "No streams configured";
        return 1;
    }

    ss::MJPEGServer server(cfg.server.url, cfg.server.port);
    server.start();

    std::vector<std::thread> workers;
    workers.reserve(streams.size());
    for (const auto& s : streams) {
        for (const auto& pr : output_profiles(s)) server.register_stream(s.id+"/"+pr.first);
        workers.emplace_back([&, s]() { run_src_worker(s, server); });
    }

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    std::cerr << "Shutting down...\n";
    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }
    server.stop();

    return 0;
}