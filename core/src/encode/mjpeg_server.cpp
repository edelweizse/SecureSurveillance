#include <encode/mjpeg_server.hpp>
#include <iostream>
#include <sstream>
#include <httplib.h>

#include "encode/webrtc_manager.hpp"

namespace ss {
    struct MJPEGServer:: Impl {
        httplib::Server svr;
    };

    MJPEGServer::MJPEGServer(std::string host, int port)
        : impl_(std::make_unique<Impl>()),
          host_(std::move(host)),
          port_(port) {}

    MJPEGServer::~MJPEGServer() {
        stop();
    }

    std::shared_ptr<MJPEGServer::StreamState> MJPEGServer::get_or_create_(const std::string& key) const {
        std::lock_guard<std::mutex> lk(streams_mtx_);
        auto& p = streams_[key];
        if (!p) p = std::make_shared<StreamState>();
        return p;
    }

    std::shared_ptr<MJPEGServer::StreamState> MJPEGServer::get_(const std::string& key) const {
        std::lock_guard<std::mutex> lk(streams_mtx_);
        auto it = streams_.find(key);
        if (it == streams_.end()) return nullptr;
        return it->second;
    }

    void MJPEGServer::push_jpeg(const std::string& stream_key,
                                std::shared_ptr<const std::vector<uint8_t>> jpeg) {
        auto st = get_or_create_(stream_key);
        {
            std::lock_guard<std::mutex> lk(st->mtx);
            st->last_jpeg = std::move(jpeg);
            ++st->seq;
        }
        st->cv.notify_all();
    }

    void MJPEGServer::push_meta(const std::string& stream_key, std::string json) {
        auto st = get_or_create_(stream_key);
        std::lock_guard<std::mutex> lk(st->meta_mtx);
        st->last_meta = std::move(json);
    }

    std::vector<std::string> MJPEGServer::list_streams() const {
        std::lock_guard<std::mutex> lk(streams_mtx_);
        std::vector<std::string> out;
        out.reserve(streams_.size());
        for (const auto& kv : streams_) out.push_back(kv.first);
        std::sort(out.begin(), out.end());
        return out;
    }

    void MJPEGServer::register_stream(const std::string& stream_key) {
        (void)get_or_create_(stream_key);
    }

    void MJPEGServer::send_webrtc(ss::WebRTCManager* w) {
        std::lock_guard lk(webrtx_mtx_);
        if (running_) {
            std::cerr << "[MJPEG] Warning: send_webrtc called after start().";
        }
        webrtc_ = w;
    }

    bool MJPEGServer::start() {
        if (running_) return true;
        running_ = true;

        impl_->svr.Post(R"(/webrtc/offer/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
            ss::WebRTCManager* webrtc = nullptr;
            {
                std::lock_guard lk(webrtx_mtx_);
                webrtc = webrtc_;
            }
            if (!webrtc) {
                res.status = 503;
                res.set_content("webrtc disabled", "text/plain");
            }
            std::string ans = webrtc->handle_offer(req.matches[1], req.body);
            if (ans.empty()) {
                res.status = 500;
                res.set_content("empty answer", "text/plain");
                return;
            }
            res.set_content(ans, "application/sdp");
        });

        impl_->svr.Get(R"(/webrtc/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
            if (req.matches.size() < 2) { res.status = 400; return; }
            const std::string stream_id = req.matches[1];
            std::string html = R"HTML(
                <!doctype html>
                    <html>
                    <head><meta charset="utf-8"><title>WebRTC Test</title></head>
                    <body>
                    <h3>WebRTC stream: )HTML" + stream_id + R"HTML(</h3>
                    <video id="v" autoplay playsinline controls style="width:90%;max-width:1100px;"></video>
                    <pre id="log"></pre>
                    <script>
                    const log = (s)=>{ document.getElementById("log").textContent += s + "\\n"; };

                    (async () => {
                      const pc = new RTCPeerConnection({ iceServers: [] }); // localhost test

                      // create client-side datachannel so server gets on-data-channel
                      const dc = pc.createDataChannel("meta");
                      dc.onmessage = (m)=>log("META: " + m.data);

                      pc.ontrack = (e) => {
                        document.getElementById("v").srcObject = e.streams[0];
                        log("ontrack: video connected");
                      };

                      const offer = await pc.createOffer({ offerToReceiveVideo: true });
                      await pc.setLocalDescription(offer);

                      const r = await fetch("/webrtc/offer/)HTML" + stream_id + R"HTML(", {
                        method: "POST",
                        headers: {"Content-Type":"application/sdp"},
                        body: offer.sdp
                      });

                      if (!r.ok) { log("offer failed: " + r.status); return; }
                      const answerSdp = await r.text();
                      await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
                      log("remote description set");
                    })();
                    </script>
                    </body>
                    </html>
            )HTML";
            res.set_content(html, "text/html; charset=utf-8");
            res.set_header("Cache-Control", "no-cache");
        });

        // /streams -> JSON list of stream keys
        impl_->svr.Get("/streams", [this](const httplib::Request&, httplib::Response& res) {
           auto keys = list_streams();
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < keys.size(); ++i) {
                oss << "\"" << keys[i] << "\"";
                if (i + 1 < keys.size()) oss << ",";
            }
            oss << "]";
            res.set_content(oss.str(), "application/json");
            res.set_header("Cache-Control", "no-cache");
        });

        // /meta/<stream_key>
        impl_->svr.Get(R"(/meta/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
            if (req.matches.size() < 2) { res.status = 400; return; }
            const std::string key = req.matches[1];

            auto st = get_(key);
            if (!st) { res.status = 404; res.set_content("{}", "application/json"); return; }

            std::string json;
            {
                std::lock_guard<std::mutex> lk(st->meta_mtx);
                json = st->last_meta.empty() ? "{}" : st->last_meta;
            }

            res.set_content(json, "application/json");
            res.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
            res.set_header("Pragma", "no-cache");
        });

        // /snapshot/<stream_key> -> last_jpeg once
        impl_->svr.Get(R"(/snapshot/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
            if (req.matches.size() < 2) { res.status = 400; return; }
            const std::string key = req.matches[1];

            auto st = get_(key);
            if (!st) { res.status = 404; return; }

            std::shared_ptr<const std::vector<uint8_t>> jpeg;
            {
                std::lock_guard<std::mutex> lk(st->mtx);
                jpeg = st->last_jpeg;
            }
            if (!jpeg || jpeg->empty()) { res.status = 204; return; }
            res.set_content(reinterpret_cast<const char *>(jpeg->data()), jpeg->size(), "image/jpeg");
            res.set_header("Cache-Control", "no-cache");
        });

        // /video/<stream_key> -> MJPEG
        impl_->svr.Get(R"(/video/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
            if (req.matches.size() < 2) { res.status = 400; return; }
            const std::string key = req.matches[1];

            auto st = get_(key);
            if (!st) { res.status = 404; return; }

            res.set_header("Cache-Control", "no-cache");
            res.set_header("Pragma", "no-cache");
            res.set_header("Connection", "close");

            const std::string boundary = "frame";
            res.set_header("Content-Type", "multipart/x-mixed-replace; boundary=" + boundary);

            res.set_chunked_content_provider(
                "multipart/x-mixed-replace; boundary=" + boundary,
                [this, st, boundary](size_t /*offset*/, httplib::DataSink& sink) {
                    uint64_t last_sent = 0;
                    {
                        std::unique_lock<std::mutex> lk(st->mtx);
                        st->cv.wait(lk, [&] { return st->seq != 0 || !running_; });
                        if (!running_) { sink.done(); return true; }
                        last_sent = st->seq;;
                    }

                    while (running_) {
                        std::shared_ptr<const std::vector<uint8_t>> jpeg;
                        uint64_t seq_local = 0;

                        {
                            std::unique_lock<std::mutex> lk(st->mtx);
                            st->cv.wait(lk, [&] { return st->seq != last_sent || !running_; });
                            if (!running_) break;

                            jpeg = st->last_jpeg;
                            seq_local = st->seq;
                        }

                        last_sent = seq_local;
                        if (!jpeg || jpeg->empty()) continue;

                        std::string header =
                            "--" + boundary + "\r\n"
                            "Content-Type: image/jpeg\r\n"
                            "Content-Length: " + std::to_string(jpeg->size()) + "\r\n\r\n";

                        if (!sink.write(header.data(), header.size())) return false;
                        if (!sink.write(reinterpret_cast<const char *>(jpeg->data()), jpeg->size())) return false;
                        if (!sink.write("\r\n", 2)) return false;
                    }

                    sink.done();
                    return true;
                }
            );
        });

        impl_->svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("ok", "text/plain");
        });

        server_thread_ = std::thread([this] {
            std::cout << "[MJPEG] Streams list: http://" << host_ << ":" << port_ << "/streams\n";
            std::cout << "[MJPEG] Video: http://" << host_ << ":" << port_ << "/video/<stream_id>/<profile>\n";
            impl_->svr.listen(host_.c_str(), port_);
        });

        return true;
    }

    void MJPEGServer::stop() {
        if (!running_) return;
        running_ = false;

        {
            std::lock_guard<std::mutex> lk(streams_mtx_);
            for (auto& kv : streams_) {
                kv.second->cv.notify_all();
            }
        }

        if (impl_) impl_->svr.stop();
        if (server_thread_.joinable()) server_thread_.join();
    }
}
