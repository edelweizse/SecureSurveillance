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
#include <chrono>
#include <thread>

#include <ingest/dual_source_factory.hpp>
#include <ingest/gst_dual_source.hpp>
#include <encode/webrtc_manager.hpp>

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

static void run_src_worker(const ss::IngestConfig& cfg, ss::MJPEGServer& server, ss::WebRTCManager& webrtc) {
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

    while (g_running) {
        std::shared_ptr<ss::FrameBundle> fb;

        if (!src->read_frame(fb, 100)) continue;
        {
            std::vector<uint8_t> enc;
            cv::imencode(".jpg", *fb->ui_bgr, enc, {cv::IMWRITE_JPEG_QUALITY, 75});
            server.push_jpeg(ui_key, std::make_shared<const std::vector<uint8_t>>(std::move(enc)));
        }

        webrtc.push_frame(fb->stream_id, *fb->ui_bgr, fb->pts_ns);
        std::string meta =
            "{"
            "\"stream_id\":\"" + ss::json_escape(fb->stream_id) + "\","
            "\"frame_id\":" + std::to_string(fb->frame_id) + ","
            "\"pts_ns\":" + std::to_string(fb->pts_ns) + ","
            "\"ui_w\":" + std::to_string(fb->ui_bgr->cols) + ","
            "\"ui_h\":" + std::to_string(fb->ui_bgr->rows) + ","
            "\"inf_w\":" + std::to_string(fb->inf_bgr->cols) + ","
            "\"inf_h\":" + std::to_string(fb->inf_bgr->rows) + ","
            "\"sx\":" + std::to_string(fb->sx) + ","
            "\"sy\":" + std::to_string(fb->sy) +
            "}";

        server.push_meta(ui_key, meta);
        server.push_meta(inf_key, meta);
        webrtc.push_meta(fb->stream_id, meta);

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

    ss::WebRTCManager webrtc;
    ss::MJPEGServer server(cfg.server.url, cfg.server.port);

    server.send_webrtc(&webrtc);
    server.start();

    std::vector<std::thread> workers;
    workers.reserve(streams.size());
    for (const auto& s : streams) {
        server.register_stream(s.id + "/ui");
        server.register_stream(s.id + "/inf");
        if (!webrtc.add_stream(s.id, s.outputs.profiles.at("ui"))) {
            std::cerr << "[main] failed to add rtc stream: " << s.id + "\n";
            continue;
        };
        workers.emplace_back([&server, &webrtc, s]() { run_src_worker(s, server, webrtc); });
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