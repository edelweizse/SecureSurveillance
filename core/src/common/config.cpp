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

    static WebcamConfig parse_webcam_config(const YAML::Node& wc) {
        WebcamConfig c;
        if (!wc) return c;
        c.device = get_str(wc, "device", "/dev/video0");
        c.width = get_int(wc, "width", 1280);
        c.height = get_int(wc, "height", 720);
        //c.fps = get_int(wc, "fps", 30);
        c.mjpg = get_bool(wc, "mjpg", get_bool(wc, "mjpeg", true));
        return c;
    }

    static FileConfig parse_file_config(const YAML::Node& fc) {
        FileConfig c;
        if (!fc) return c;
        c.path = get_str(fc, "path", "/");
        c.fps = get_int(fc, "fps", 30);
        c.loop = get_bool(fc, "loop", false);
        return c;
    }

    static RTSPConfig parse_rtsp_config(const YAML::Node& rc) {
        RTSPConfig c;
        if (!rc) return c;
        c.url = get_str(rc, "url", "/");
        //c.fps = get_int(rc, "fps", 30);
        c.latency_ms = get_int(rc, "latency_ms", 1000);
        c.tcp = get_bool(rc, "tcp", true);
        return c;
    }

    static ReplicateConfig parse_replicate_config(const YAML::Node& r) {
        ReplicateConfig c;
        if (!r) return c;
        c.count = get_int(r, "count", 1);
        if (r["ids"]) c.ids = r["ids"].as<std::vector<std::string>>();
        if (c.count < 1) c.count = 1;
        return c;
    }

    static OutputConfig parse_output_config(const YAML::Node& o, const OutputConfig& def = {}) {
        OutputConfig c = def;
        if (!o) return c;
        c.width = get_int(o, "width", c.width);
        c.height = get_int(o, "height", c.height);
        c.fps = get_int(o, "fps", c.fps);
        c.format = get_str(o, "format", c.format);
        c.keep_aspect = get_bool(o, "keep_aspect", c.keep_aspect);
        c.interp = get_str(o, "interp", c.interp);
        c.jpeg_quality = get_int(o, "jpeg_quality", c.jpeg_quality);
        return c;
    }

    static OutputsConfig parse_outputs_config(const YAML::Node& o) {
        OutputsConfig out;
        if (!o) return out;
        auto profiles = o["profiles"];
        if (!profiles) return out;
        if (!profiles.IsMap()) {
            throw std::runtime_error ("[Config] profiles must be a map!");
        }

        for (auto it = profiles.begin(); it != profiles.end(); ++it) {
            const auto name = it->first.as<std::string>();
            const YAML::Node cfg = it->second;
            out.profiles[name] = parse_output_config(cfg, OutputConfig{});
        }
        return out;
    }

    AppConfig load_config_yaml(const std::string& path) {
        AppConfig cfg;
        YAML::Node root = YAML::LoadFile(path);

        const YAML::Node srv = root["server"];
        cfg.server.url = get_str(srv, "host", "0.0.0.0");
        cfg.server.port = get_int(srv, "port", 8080);

        auto arr = root["streams"];
        if (!arr || !arr.IsSequence() || arr.size() <= 0) {
            throw std::runtime_error ("[Config] no streams specified!");
        }

        for (const auto& s : arr) {
            IngestConfig ic;
            ic.id = get_str(s, "id", "unk");
            ic.type = get_str(s, "type", "unk");

            ic.webcam = parse_webcam_config(s["webcam"]);
            ic.file = parse_file_config(s["file"]);
            ic.rtsp = parse_rtsp_config(s["rtsp"]);

            ic.replicate = parse_replicate_config(s["replicate"]);
            ic.output = parse_output_config(s["output"], OutputConfig{});
            ic.outputs = parse_outputs_config(s["outputs"]);

            if (ic.type == "rtsp" && ic.rtsp.url.empty()) {
                throw std::runtime_error ("[Config] RTSP stream " + ic.id + " has empty URL!");
            }

            cfg.streams.push_back(std::move(ic));
        }
        return cfg;
    }
}
