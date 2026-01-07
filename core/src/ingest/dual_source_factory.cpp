#include <ingest/dual_source_factory.hpp>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <gst/gstutils.h>

namespace ss {

    static std::string to_file_uri(const std::string& path) {
        namespace fs = std::filesystem;
        GError* err = nullptr;
        if (!fs::exists(path)) {
            throw std::runtime_error("path not found: " + path);
        }

        std::string abs = fs::absolute(path).string();
        gchar* uri = gst_filename_to_uri(abs.c_str(), &err);
        if (!uri) {
            std::string msg = "Failed to convert filename to URI: ";
            if (err) {
                msg += err->message;
                g_error_free(err);
            }
            throw std::runtime_error(msg);
        }

        std::string res(uri);
        g_free(uri);
        return res;
    }

    static OutputConfig need_profile(const IngestConfig& cfg, const std::string& name) {
        auto it = cfg.outputs.profiles.find(name);
        if (it == cfg.outputs.profiles.end()) {
            throw std::runtime_error ("missing output profile " + name + " for stream " + cfg.id + ".\n");
        }
        return it->second;
    }

    static std::string caps(const OutputConfig& o) {
        std::ostringstream ss;
        ss << "video/x-raw,format=" << (o.format.empty() ? "BGR" : o.format);
        if (o.width > 0 && o.height > 0) ss << ",width=" << o.width << ",height=" << o.height;
        if (o.fps > 0) ss << ",framerate=" << o.fps << "/1";
        return ss.str();
    }

    static std::string make_queue(bool is_live) {
        if (is_live) {
            return "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0";
        }
        return "queue max-size-buffers=5";
    }

    static std::string common_split_tail(const std::string& sink_inf,
                                         const std::string& sink_ui,
                                         const OutputConfig& inf,
                                         const OutputConfig& ui,
                                         bool is_live) {
        std::ostringstream ss;
        std::string sync_str = is_live ? "sync=false" : "sync=true";
        std::string queue_str = make_queue(is_live);
        ss
            << " ! tee name=t "

            << "t. ! " << queue_str
            << " ! videorate "
            << " ! videoscale "
            << " ! videoconvert "
            << " ! " << caps(inf)
            << " ! appsink name=" << sink_inf << " max-buffers=1 drop=true " << sync_str << " "

            << "t. ! " << queue_str
            << " ! videorate "
            << " ! videoscale "
            << " ! videoconvert "
            << " ! " << caps(ui)
            << " ! appsink name=" << sink_ui << " max-buffers=1 drop=true " << sync_str << " ";
        return ss.str();
    }

    static std::string file_dual_pipeline(const IngestConfig& cfg,
                                          const std::string& sink_inf,
                                          const std::string& sink_ui,
                                          const OutputConfig& inf,
                                          const OutputConfig& ui) {
        std::ostringstream ss;
        const std::string uri = to_file_uri(cfg.file.path);
        ss << "uridecodebin uri=\"" << uri << "\" name=d "
           << "d. ! videoconvert ! video/x-raw ";
        ss << common_split_tail(sink_inf, sink_ui, inf, ui, false);
        return ss.str();
    }

    static std::string rtsp_dual_pipeline(const IngestConfig& cfg,
                                          const std::string& sink_inf,
                                          const std::string& sink_ui,
                                          const OutputConfig& inf,
                                          const OutputConfig& ui) {
        const auto& r = cfg.rtsp;
        std::string proto = r.tcp ? "tcp" : "udp";

        std::ostringstream ss;
        ss << "rtspsrc location=\"" << r.url << "\" latency=" << r.latency_ms
           << " protocols=" << proto << " drop-on-latency=true "
           << " ! decodebin ! videoconvert ! video/x-raw ";
        ss << common_split_tail(sink_inf, sink_ui, inf, ui, true);
        return ss.str();
    }

    static std::string webcam_dual_pipeline(const IngestConfig& cfg,
                                            const std::string& sink_inf,
                                            const std::string& sink_ui,
                                            const OutputConfig& inf,
                                            const OutputConfig& ui) {
        const auto& w = cfg.webcam;
        std::ostringstream ss;
        if (w.mjpg) {
            ss << "v4l2src device=" << w.device << " ! "
               << "image/jpeg,width=" << w.width << ",height=" << w.height
               << ",framerate=30/1 ! jpegdec ";
        } else {
            ss << "v4l2src device=" << w.device << " ! "
               << "video/x-raw,width=" << w.width << ",height=" << w.height
               << ",framerate=30/1 ";
        }
        ss << common_split_tail(sink_inf, sink_ui, inf, ui, true);
        return ss.str();
    }

    std::unique_ptr<GstDualSource> make_dual_source(const IngestConfig& cfg) {
        OutputConfig inf = need_profile(cfg, "inference");
        OutputConfig ui = need_profile(cfg, "ui");

        const std::string sink_inf = "sink_" + cfg.id + "_inf";
        const std::string sink_ui = "sink_" + cfg.id + "_ui";
        std::string pipe;

        if (cfg.type == "file") pipe = file_dual_pipeline(cfg, sink_inf, sink_ui, inf, ui);
        else if (cfg.type == "webcam") pipe = webcam_dual_pipeline(cfg, sink_inf, sink_ui, inf, ui);
        else if (cfg.type == "rtsp") pipe = rtsp_dual_pipeline(cfg, sink_inf, sink_ui, inf, ui);
        else throw std::invalid_argument("[Config] Unknown source type: " + cfg.type + ".\n");

        return std::make_unique<GstDualSource>(pipe, cfg.id, sink_inf, sink_ui);
    }
}
