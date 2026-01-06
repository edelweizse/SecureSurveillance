#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace ss {
    struct WebcamConfig {
        std::string device;
        int width;
        int height;
        //int fps;
        bool mjpg;
    };

    struct FileConfig {
        std::string path;
        int fps;
        bool loop;
    };

    struct RTSPConfig {
        std::string url;
        int latency_ms;
        //int fps;
        bool tcp;
    };

    struct ReplicateConfig {
        int count = 1;
        std::vector<std::string> ids;
    };

    struct OutputConfig {
        int width = 0;
        int height = 0;
        int fps = 0;
        bool keep_aspect = true;
        std::string interp = "linear"; // nearest|cubic|linear|area
        std::string format = "BGR";
        int jpeg_quality = 75;
    };

    struct OutputsConfig {
        std::unordered_map<std::string, OutputConfig> profiles;
    };

    struct IngestConfig {
        std::string type; // webcam|file|rtsp
        std::string id;

        WebcamConfig webcam;
        FileConfig file;
        RTSPConfig rtsp;

        ReplicateConfig replicate;

        OutputConfig output;

        OutputsConfig outputs;
    };

    struct ServerConfig {
        std::string url;
        int port;
    };

    struct AppConfig {
        ServerConfig server;
        std::vector<IngestConfig> streams;
    };

    AppConfig load_config_yaml(const std::string& path);
}