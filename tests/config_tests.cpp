#include <common/config.hpp>
#include <common/replicate.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    int g_failures = 0;

    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++g_failures;
            std::cerr << "[FAIL] " << message << "\n";
        }
    }

    std::string write_yaml_file(const std::string& prefix, const std::string& body) {
        namespace fs = std::filesystem;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const fs::path path = fs::temp_directory_path() /
                              (prefix + "_" + std::to_string(stamp) + ".yaml");

        std::ofstream out(path);
        if (!out.is_open()) {
            throw std::runtime_error("failed to open temp config file: " + path.string());
        }
        out << body;
        out.close();
        return path.string();
    }

    bool load_throws(const std::string& yaml) {
        const std::string path = write_yaml_file("ss_cfg", yaml);
        try {
            (void)ss::load_config_yaml(path);
            std::filesystem::remove(path);
            return false;
        } catch (...) {
            std::filesystem::remove(path);
            return true;
        }
    }

    void test_expand_replicas_fills_missing_ids() {
        ss::IngestConfig in_cfg;
        in_cfg.id = "cam0";
        in_cfg.type = "webcam";
        in_cfg.replicate.count = 3;
        in_cfg.replicate.ids = {"custom_0"};

        const std::vector<ss::IngestConfig> input = {in_cfg};
        const auto expanded = ss::expand_replicas(input);

        check(expanded.size() == 3, "expand_replicas should output replicate.count entries");
        check(expanded[0].id == "custom_0", "expand_replicas should preserve provided ids");
        check(expanded[1].id == "cam0_1", "expand_replicas should synthesize missing id #1");
        check(expanded[2].id == "cam0_2", "expand_replicas should synthesize missing id #2");
    }

    void test_config_rejects_legacy_output() {
        const std::string yaml =
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    output:\n"
            "      width: 1280\n"
            "      height: 720\n";

        check(load_throws(yaml), "load_config_yaml should reject legacy stream.output schema");
    }

    void test_config_requires_global_outputs_fps() {
        const std::string yaml =
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    outputs:\n"
            "      profiles:\n"
            "        inference:\n"
            "          width: 640\n"
            "          height: 640\n"
            "        ui:\n"
            "          width: 1280\n"
            "          height: 720\n";

        check(load_throws(yaml), "load_config_yaml should require outputs.fps > 0");
    }

    void test_global_outputs_fps_overrides_profile_fps() {
        const std::string yaml =
            "server:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 8080\n"
            "streams:\n"
            "  - id: \"file0\"\n"
            "    type: \"file\"\n"
            "    file:\n"
            "      path: \"/tmp/test.mp4\"\n"
            "    outputs:\n"
            "      fps: 12\n"
            "      profiles:\n"
            "        inference:\n"
            "          width: 640\n"
            "          height: 640\n"
            "          fps: 5\n"
            "        ui:\n"
            "          width: 1280\n"
            "          height: 720\n"
            "          fps: 30\n";

        const std::string path = write_yaml_file("ss_cfg_ok", yaml);
        const auto cfg = ss::load_config_yaml(path);
        std::filesystem::remove(path);

        check(cfg.streams.size() == 1, "valid config should load exactly one stream");
        const auto& profiles = cfg.streams[0].outputs.profiles;
        check(cfg.streams[0].outputs.fps == 12, "outputs.fps should be stored");
        check(profiles.at("inference").fps == 12, "inference fps should be synchronized to outputs.fps");
        check(profiles.at("ui").fps == 12, "ui fps should be synchronized to outputs.fps");
    }
}

int main() {
    test_expand_replicas_fills_missing_ids();
    test_config_rejects_legacy_output();
    test_config_requires_global_outputs_fps();
    test_global_outputs_fps_overrides_profile_fps();

    if (g_failures != 0) {
        std::cerr << "[FAIL] total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "[OK] all config tests passed\n";
    return 0;
}
