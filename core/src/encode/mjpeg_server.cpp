#include <encode/mjpeg_server.hpp>
#include <iostream>
#include <httplib.h>

namespace ss {
    struct MJPEGServer:: Impl {
        httplib::Server svr;
    };

    MJPEGServer::MJPEGServer(std::string& host, int port)
        : impl_(new Impl()), host_(host), port_(port) {}

    MJPEGServer::~MJPEGServer() {
        stop();
        delete impl_;
    }

    void MJPEGServer::push_jpeg(std::vector<uint8_t>&& jpeg) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            last_jpeg_ = std::move(jpeg);
            ++seq_;
        }
        cv_.notify_all();
    }

    void MJPEGServer::push_meta(std::string&& json) {
        std::lock_guard<std::mutex> lk(meta_mtx_);
        last_meta_ = std::move(json);
    }

    bool MJPEGServer::start() {
        if (running_) return true;
        running_ = true;

        // MJPEG endpoint
        impl_->svr.Get("/video", [this](const httplib::Request&, httplib::Response& res) {
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Pragma", "no-cache");
            res.set_header("Connection", "close");

            const std::string boundary = "frame";
            res.set_header("Content-Type", "multipart/x-mixed-replace; boundary=" + boundary);

            res.set_chunked_content_provider(
                "multipart/x-mixed-replace; boundary=" + boundary,
                [this, boundary](size_t /* offset */, httplib::DataSink& sink) {
                    uint64_t last_sent = 0;
                    // if no frame wait some
                    {
                        std::unique_lock<std::mutex> lk(mtx_);
                        cv_.wait(lk, [&]{ return seq_ > 0 || !running_; });
                        if (!running_) {sink.done(); return true;}
                        last_sent = seq_;
                    }

                    while (running_) {
                        std::vector<uint8_t> jpeg;
                        uint64_t seq_local = 0;

                        {
                            std::unique_lock<std::mutex> lk(mtx_);
                            cv_.wait(lk, [&] { return seq_ != last_sent || !running_; });
                            if (!running_) break;
                            jpeg = last_jpeg_;
                            seq_local = seq_;
                        }
                        last_sent = seq_local;
                        if (jpeg.empty()) continue;

                        std::string header =
                            "--" + boundary + "\r\n"
                            "Content-Type: image/jpeg\r\n"
                            "Content-Length: " + std::to_string(jpeg.size()) + "\r\n\r\n";
                        if (!sink.write(header.data(), header.size())) return false;
                        if (!sink.write(reinterpret_cast<const char*>(jpeg.data()), jpeg.size())) return false;
                        if (!sink.write("\r\n", 2)) return false;
                    }
                    sink.done();
                    return true;
                }
            );
        });

        impl_->svr.Get("/meta", [this](const httplib::Request&, httplib::Response& res) {
            std::string json;
            {
                std::lock_guard<std::mutex> lk(meta_mtx_);
                json = last_meta_;
            }
            res.set_content(json, "application/json");
            res.set_header("Cache-Control", "no-cache");
        });

        impl_->svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("ok", "text/plain");
        });

        server_thread_ = std::thread([this] {
            std::cout << "[MJPEG] Listening on http://" << host_ << ":" << port_ << "\n";
            std::cout << "[MJPEG] Video stream on http://" << host_ << ":" << port_ << "/video" << "\n";
            impl_->svr.listen(host_.c_str(), port_);
        });

        return true;
    }

    void MJPEGServer::stop() {
        if (!running_) return;
        running_ = false;
        cv_.notify_all();
        if (impl_) impl_->svr.stop();
        if (server_thread_.joinable()) server_thread_.join();
    }
}
