#pragma once

#include <string>
#include <opencv2/core.hpp>

namespace ss {
    struct OutputConfig;

    class WebRTCManager {
    public:
        WebRTCManager();
        ~WebRTCManager();

        WebRTCManager(const WebRTCManager&) = delete;
        WebRTCManager& operator=(const WebRTCManager&) = delete;

        bool add_stream(const std::string& stream_id, const OutputConfig& config);

        void push_frame(const std::string& stream_id,
                        const cv::Mat& frame,
                        int64_t pts_ns);

        void push_meta(const std::string& stream_id,
                       const std::string& meta);

        std::string handle_offer(const std::string& stream_id,
                                 const std::string& sdp_offer);
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}