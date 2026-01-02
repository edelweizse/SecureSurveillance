#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ss {
    class MJPEGServer {
    public:
        MJPEGServer(std::string& host, int port);
        ~MJPEGServer();

        // Start http server in bg thread
        bool start();
        void stop();

        // push latest jpeg frame in bytes
        void push_jpeg(std::vector<uint8_t>&& jpeg);

        // optionally push json metadata
        void push_meta(std::string&& json);
    private:
        struct Impl;
        Impl* impl_;
        std::string host_;
        int port_;

        std::thread server_thread_;
        std::atomic<bool> running_{false};

        std::mutex mtx_;
        std::condition_variable cv_;
        std::vector<uint8_t> last_jpeg_;
        uint64_t seq_{0};

        std::mutex meta_mtx_;
        std::string last_meta_ = "{}";
    };
}