#include <common/config.hpp>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace ss {
    static bool get_bool(
        const YAML::Node& n, const char* key, bool def) {
        return (n && n[key]) ? n[key].as<bool>() : def;
    }

    static int get_int(
        const YAML::Node& n, const char* key, int def) {
        return (n && n[key]) ? n[key].as<int>() : def;
    }

    static std::string get_str(
        const YAML::Node& n, const char* key, const std::string& def) {
        return (n && n[key]) ? n[key].as<std::string>() : def;
    }

    AppConfig load_config_yaml(const std::string& path) {
        AppConfig cfg;
        YAML::Node root = YAML::LoadFile(path);

        auto ingest = root["ingest"];
        cfg.ingest.type = get_str(ingest, "type", "webcam");
        cfg.ingest.src_id = get_str(ingest, "src_id", "cam0");

        // webcam
        auto wc = ingest["webcam"];
        cfg.ingest.webcam.device = get_str(wc, "device", "/dev/video0");
        cfg.ingest.webcam.width = get_int(wc, "width", 1280);
        cfg.ingest.webcam.height = get_int(wc, "height", 720);
        cfg.ingest.webcam.fps = get_int(wc, "fps", 30);
        cfg.ingest.webcam.mjpg = get_bool(wc, "mjpg", true);

        auto fc = ingest["file"];
        cfg.ingest.file.path = get_str(fc, "path", "assets/test_video.mp4");
        cfg.ingest.file.loop = get_bool(fc, "loop", true);

        auto rc = ingest["rtsp"];
        cfg.ingest.rtsp.url = get_str(rc, "url", "http://www.rtsp.url");
        cfg.ingest.rtsp.latency_ms = get_int(rc, "latency_ms", 100);
        cfg.ingest.rtsp.tcp = get_bool(rc, "tcp", true);

        return cfg;
    }
}
