#include <ingest/gst_frame_source.hpp>
#include <common/config.hpp>
#include <encode/mjpeg_server.hpp>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <yaml-cpp/exceptions.h>

#include "ingest/frame_source_factory.hpp"

static void print_help() {
    std::cerr << "Usage: ss_core_service <path/to/config.yaml>\n";
}

int main(int argc, char** argv) {
    std::string cfg_path = "../../../configs/webcam.yaml";
    if (argc >= 2) cfg_path = argv[1];
    else {
        std::cerr << "Using default config" << cfg_path << "\n";
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

    cv::namedWindow("Ingest Debug", cv::WINDOW_NORMAL);
    cv::resizeWindow("Ingest Debug",
                     cfg.ingest.webcam.width > 0 ? cfg.ingest.webcam.width : 1280,
                     cfg.ingest.webcam.height > 0 ? cfg.ingest.webcam.height : 720);
    ss::FramePacket fp;

    int empty_count = 0;
    while (true) {
        if (!src->read(fp, 1000)) {
            if (++empty_count % 60 == 0)
                std::cerr << "No frames yet..\n";
            continue;
        }
        empty_count = 0;

        if (fp.bgr.empty()) {
            std::cerr << "Got empty frame\n";
            continue;
        };

        cv::imshow("Ingest Debug", fp.bgr);

        std::vector<uint8_t> jpeg;
        cv::imencode(".jpg", fp.bgr, jpeg, jpeg_params);
        server.push_jpeg(std::move(jpeg));
        server.push_meta(
            std::string("{\"source\":\"") + cfg.ingest.src_id + "\",\"frame_id\":" + std::to_string(fp.frame_id) + "}"
        );

        if (cv::waitKey(1) == 27) break;
    }
    src->stop();
    server.stop();
    return 0;
}