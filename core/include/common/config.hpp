#pragma once

#include <string>

namespace ss {
    struct WebConfig {
        std::string device = "/dev/video0";
        int width = 1280;
        int height = 720;
        int fps = 30;
        bool mjpg = true;
    };

    struct FileConfig {
        std::string path = "assets/test_video.mp4";
        bool loop = true;
    };

    struct RTSPConfig {
        std::string url;
        int latency_ms = 100;
        bool tcp = true;
    };

    struct IngestConfig {
        std::string type = "web"; // web | file | rtsp
        std::string src_id = "cam0";
        WebConfig webcam;
        FileConfig file;
        RTSPConfig rtsp;
    };

    struct AppConfig {
        IngestConfig ingest;
    };

    AppConfig load_config_yaml(const std::string& path);
}