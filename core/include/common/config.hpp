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

    struct IngestConfig {
        std::string type = "web";
        std::string src_id = "cam0";
        WebConfig webcam;
    };

    struct AppConfig {
        IngestConfig ingest;
    };

    AppConfig load_config_yaml(const std::string& path);
}