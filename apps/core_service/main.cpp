#include <ingest/gst_frame_source.hpp>
#include <common/config.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

static std::string webcam_pipeline(const ss::WebConfig& c) {
    if (c.mjpg) {
        return "v4l2src device=" + c.device + " ! "
               "image/jpeg,width=" + std::to_string(c.width)
               + ",height=" + std::to_string(c.height)
               + ",framerate=" + std::to_string(c.fps) + "/1 ! "
               "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
               "appsink name=sink max-buffers=2 drop=true sync=false";
    } else {
        return "v4l2src device=" + c.device + " ! "
               "video/x-raw,width=" + std::to_string(c.width)
               + ",height=" + std::to_string(c.height)
               + ",framerate=" + std::to_string(c.fps) + "/1 ! "
               "videoconvert ! video/x-raw,format=BGR ! "
               "appsink name=sink max-buffers=2 drop=true sync=false";
    }
}

int main(int argc, char** argv) {
    std::string cfg_path = "../../../configs/default.yaml";
    if (argc >= 2) cfg_path = argv[1];

    ss::AppConfig cfg = ss::load_config_yaml(cfg_path);
    if (cfg.ingest.type != "webcam") {
        std::cerr << "Only webcam supported\n";
        return 1;
    }

    std::string pipeline = webcam_pipeline(cfg.ingest.webcam);
    std::cout << "Config: " + cfg_path + "\n" << "Using pipeline:\n" << pipeline << "\n";

    ss::GstFrameSource src(pipeline, cfg.ingest.src_id);
    if (!src.start()) {
        std::cerr << "Failed to start GstFrameSource.\n";
        return 1;
    }

    ss::FramePacket fp;
    cv::namedWindow("Ingest Debug", cv::WINDOW_NORMAL);
    cv::resizeWindow("Ingest Debug", 1280, 720);

    int empty_count = 0;
    while (true) {
        if (!src.read(fp, 1000)) {
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
        if (cv::waitKey(1) == 27) break;
    }
    src.stop();
    return 0;
}