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

#include "pipeline/runtime.hpp"

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

    for (const auto& s : streams) {
        server.register_stream(s.id + "/ui");
        server.register_stream(s.id + "/inf");
    }

    ss::PipelineRuntime::Options opt;
    opt.jpeg_quality = 75;
    opt.inf_workers = 4;

    ss::PipelineRuntime rt(server, streams, opt);
    rt.start();

    while (g_running) std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cerr << "Shutting down...\n";

    rt.stop();
    server.stop();
    return 0;
}