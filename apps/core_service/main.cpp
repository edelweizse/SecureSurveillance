#include <common/config.hpp>
#include <encode/mjpeg_server.hpp>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/exceptions.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <csignal>

#include "ingest/dual_source_factory.hpp"
#include "ingest/gst_dual_source.hpp"

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

static void run_src_worker(const ss::IngestConfig& cfg, ss::MJPEGServer& server) {
    std::unique_ptr<ss::GstDualSource> src;
    try {
        src = ss::make_dual_source(cfg);
    } catch (const std::exception& e) {
        std::cerr << "[worker:" << cfg.id << "] Failed to create dual source: " << e.what() << "\n";
        return;
    }

    if (!src->start()) {
        std::cerr << "[worker:" << cfg.id << "] Failed to start dual source.\n";
        return;
    }

    const std::string inf_key = cfg.id + "/inf";
    const std::string ui_key = cfg.id + "/ui";

    ss::FramePacket fp;
    ss::JpegPacket jp;

    while (g_running) {
        if (src->read_ui(jp, 100)) {
            server.push_jpeg(ui_key, jp.jpeg);

            std::string meta =
                "{"
                "\"stream_id\":\"" + cfg.id + "\","
                "\"profile\":\"ui\","
                "\"frame_id\":" + std::to_string(jp.frame_id) + ","
                "\"pts_ns\":" + std::to_string(jp.pts_ns) +
                "}";
            server.push_meta(ui_key, std::move(meta));
        }
        if (src->read_inference(fp, 100)) {
            std::string meta =
                "{"
                "\"stream_id\":\"" + cfg.id + "\","
                "\"profile\":\"inf\","
                "\"frame_id\":" + std::to_string(fp.frame_id) + ","
                "\"pts_ns\":" + std::to_string(fp.pts_ns) + ","
                "\"w\":" + std::to_string(fp.bgr.cols) + ","
                "\"h\":" + std::to_string(fp.bgr.rows) +
                "}";
            server.push_meta(inf_key, std::move(meta));

        }
    }

    src->stop();
    std::cerr << "[worker:" << cfg.id << "] Stopped.\n";
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    std::string cfg_path = "../../../configs/dual.yaml";
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
        server.register_stream(s.id + "/ui");
        server.register_stream(s.id + "/inf");
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