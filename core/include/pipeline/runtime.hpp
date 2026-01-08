#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <common/config.hpp>
#include <encode/mjpeg_server.hpp>
#include <ingest/gst_dual_source.hpp>

#include <pipeline/types.hpp>
#include <pipeline/bounded_queue.hpp>

namespace ss {
    class PipelineRuntime {
    public:
        struct Options {
            int jpeg_quality = 75;

            // queue capacities
            size_t infer_in_cap = 50; // global
            size_t anon_in_cap = 5; // per stream
            size_t res_cap = 20; // per stream
            size_t enc_in_cap = 5; // per stream

            int inf_workers = 4; // M
        };

        PipelineRuntime(MJPEGServer& server,
                        std::vector<IngestConfig> streams,
                        Options opt);

        bool start();
        void stop();

        ~PipelineRuntime() { stop(); }
    private:
        struct StreamPipe {
            std::string stream_id;

            BoundedQueue<FramePtr> anon_in;
            BoundedQueue<InferResults> inf_res;
            BoundedQueue<FramePtr> enc_in;

            std::thread ingest_thr;
            std::thread enc_thr;
            std::thread anon_thr;

            StreamPipe(std::string id, size_t anon_cap, size_t res_cap, size_t enc_cap)
                : stream_id(std::move(id)),
                  anon_in(anon_cap),
                  inf_res(res_cap),
                  enc_in(enc_cap) {}
        };

        // workers
        void ingest_loop_(const IngestConfig& cfg, std::unique_ptr<GstDualSource> src, StreamPipe* pipe);
        void infer_loop_();
        void anonymizer_loop_(StreamPipe* pipe);
        void encoder_loop_(StreamPipe* pipe);

        // modules
        std::vector<Box> run_inference_(const cv::Mat& inf_frame);
        void anonymize_(cv::Mat& ui_frame, const std::vector<Box>& boxes, float sx, float sy);


        MJPEGServer& server_;
        std::vector<IngestConfig> streams_;
        Options opt_;

        std::atomic<bool> running_{false};

        // global inference input
        BoundedQueue<FramePtr> infer_in_;

        // per stream pipes
        std::vector<std::unique_ptr<StreamPipe>> pipes_;

        // stream_id -> pipe*
        std::unordered_map<std::string, StreamPipe*> pipes_by_stream_id_;

        // pool threads
        std::vector<std::thread> infer_pool_;
    };
}