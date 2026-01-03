#include <ingest/gst_frame_source.hpp>
#include <ingest/frame_source_factory.hpp>
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

static std::atomic<bool> g_running(true);
static void handle_sigint(int) {g_running = false;}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    std::string cfg_path = "../../../configs/rtsp.yaml";
    if (argc >= 2) {
        cfg_path = argv[1];
    } else {
        std::cerr << "Using default config: " << cfg_path << "\n";
    }

    ss::AppConfig cfg;
    try {
        cfg = ss::load_config_yaml(cfg_path);
    } catch (const YAML::Exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    std::unique_ptr<ss::IFrameSource> src;
    try {
        src = ss::make_frame_source(cfg.ingest);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create frame src: " << e.what() << "\n";
        return 1;
    }
    if (!src->start()) {
        std::cerr << "Failed to start source.\n";
        return 1;
    }

    std::string host = "0.0.0.0";
    ss::MJPEGServer server(host, 8080);
    server.start();

    std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, 100};

    ss::FramePacket fp;

    int empty_count = 0;
    while (g_running) {
        if (!src->read(fp, 5000)) {
            if (++empty_count % 60 == 0)
                std::cerr << "No frames yet..\n";
            continue;
        }
        empty_count = 0;

        if (fp.bgr.empty()) {
            std::cerr << "Got empty frame\n";
            continue;
        };

        std::vector<uint8_t> jpeg;
        cv::imencode(".jpg", fp.bgr, jpeg, jpeg_params);
        server.push_jpeg(std::move(jpeg));
        server.push_meta(
            std::string(R"({"source":")") + cfg.ingest.src_id + R"(","frame_id":)" + std::to_string(fp.frame_id) + "}"
        );

        if (cv::waitKey(1) == 27) g_running = false;
    }
    std::cerr << "Shutting down...\n";
    src->stop();
    server.stop();
    return 0;
}