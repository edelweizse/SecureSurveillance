#include <ingest/frame_source_factory.hpp>
#include <ingest/gst_frame_source.hpp>

namespace ss {
    static std::string web_pipeline(const WebcamConfig& c, const std::string& sink_name) {
        if (c.mjpg) {
            return "v4l2src device=" + c.device + " ! "
                   "image/jpeg,width=" + std::to_string(c.width)
                   + ",height=" + std::to_string(c.height)
                   + ",framerate=" + std::to_string(c.fps) + "/1 ! "
                   "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
                   "appsink name=" + sink_name + " max-buffers=2 drop=true sync=false";
        }
        return "v4l2src device=" + c.device + " ! "
               "video/x-raw,width=" + std::to_string(c.width)
               + ",height=" + std::to_string(c.height)
               + ",framerate=" + std::to_string(c.fps) + "/1 ! "
               "videoconvert ! video/x-raw,format=BGR ! "
               "appsink name=" + sink_name + " max-buffers=2 drop=true sync=false";
    }

    static std::string file_pipeline(const FileConfig& c, const std::string& sink_name) {
        return "filesrc location=\"" + c.path + "\" ! "
               "decodebin ! videoconvert ! video/x-raw,format=BGR ! "
               "appsink name=" + sink_name + " max-buffers=2 drop=true sync=false";
    }

    static std::string rtsp_pipeline(const RTSPConfig& c, const std::string& sink_name) {
        std::string proto = c.tcp ? "tcp" : "udp";
        return "rtspsrc location=\"" + c.url +
               "\" latency=" + std::to_string(c.latency_ms) +
               " protocols=" + proto + " drop-on-latency=true ! "
               "decodebin ! videoconvert ! video/x-raw,format=BGR ! "
               "appsink name=" + sink_name + " max-buffers=2 drop=true sync=false";
    }

    std::unique_ptr<IFrameSource> make_frame_source(const IngestConfig& cfg) {
        std::string pipe;
        const std::string sink_name = "sink_" + cfg.id;

        if (cfg.type == "webcam") {
            pipe = web_pipeline(cfg.webcam, sink_name);
        } else if (cfg.type == "file") {
            pipe = file_pipeline(cfg.file, sink_name);
        } else if (cfg.type == "rtsp") {
            if (cfg.rtsp.url.empty()) {
                throw std::runtime_error("rtsp.url is empty in config");
            }
            pipe = rtsp_pipeline(cfg.rtsp, sink_name);
        } else {
            throw std::runtime_error("Unknown source type" + cfg.type);
        }

        return std::make_unique<GstFrameSource>(pipe, cfg.id, sink_name);
    }
}
