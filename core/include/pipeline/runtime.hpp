#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <anonymization/anonymizer.hpp>
#include <common/config.hpp>
#include <encode/mjpeg_server.hpp>
#include <inference/yunet_detector.hpp>
#include <ingest/gst_dual_source.hpp>
#include <pipeline/bounded_queue.hpp>
#include <pipeline/types.hpp>

namespace ss {
    class PipelineRuntime {
    public:
        struct Options {
            int jpeg_quality = 75;

            // queue capacities
            size_t infer_in_cap = 50; // global detector pool input
            size_t inf_state_in_cap = 5; // per-stream ordered inference state input
            size_t det_res_cap = 20; // per-stream detector results
            size_t anon_in_cap = 5; // per-stream tracked frame input
            size_t enc_in_cap = 5; // per-stream encoder input
            size_t analytics_cap = 256; // global tracker output

            int inf_workers = 1;

            std::string detector_param_path = "models/detector/face_detection_yunet_2023mar.ncnn.param";
            std::string detector_bin_path = "models/detector/face_detection_yunet_2023mar.ncnn.bin";
            int detector_input_w = 640;
            int detector_input_h = 640;
            float detector_score_thresh = 0.6f;
            float detector_nms_thresh = 0.3f;
            int detector_top_k = 750;
            int detector_ncnn_threads = 1;

            std::string anonymizer_method = "pixelate";
            int anonymizer_pixelation_divisor = 10;
            int anonymizer_blur_kernel = 31;

            float tracker_high_thresh = 0.6f;
            float tracker_low_thresh = 0.2f;
            float tracker_match_iou_thresh = 0.3f;
            float tracker_low_match_iou_thresh = 0.2f;
            int tracker_min_hits = 2;
            int tracker_max_missed = 20;
        };

        PipelineRuntime(MJPEGServer& server,
                        std::vector<IngestConfig> streams,
                        Options opt);

        bool start();
        void stop();
        bool pop_tracker_output(TrackerFrameOutput& out, std::chrono::milliseconds timeout);

        ~PipelineRuntime() { stop(); }

    private:
        struct StreamPipe {
            std::string stream_id;

            BoundedQueue<FramePtr> inf_state_in;
            BoundedQueue<InferResults> det_res;
            BoundedQueue<FramePtr> anon_in;
            BoundedQueue<FramePtr> enc_in;

            std::thread ingest_thr;
            std::thread inf_state_thr;
            std::thread anon_thr;
            std::thread enc_thr;

            StreamPipe(std::string id,
                       size_t inf_state_cap,
                       size_t det_res_queue_cap,
                       size_t anon_cap,
                       size_t enc_cap)
                : stream_id(std::move(id)),
                  inf_state_in(inf_state_cap),
                  det_res(det_res_queue_cap),
                  anon_in(anon_cap),
                  enc_in(enc_cap) {}
        };

        // workers
        void ingest_loop_(const IngestConfig& cfg, std::unique_ptr<GstDualSource> src, StreamPipe* pipe);
        void infer_loop_();
        void infer_state_loop_(StreamPipe* pipe);
        void anonymizer_loop_(StreamPipe* pipe);
        void encoder_loop_(StreamPipe* pipe);

        // modules
        std::vector<Box> run_inference_(const cv::Mat& inf_frame);
        void anonymize_(cv::Mat& ui_frame,
                        const std::vector<Box>& boxes,
                        float sx,
                        float sy,
                        float tx,
                        float ty);
        void draw_tracks_(cv::Mat& ui_frame,
                          const std::vector<Box>& boxes,
                          float sx,
                          float sy,
                          float tx,
                          float ty);
        void publish_tracker_output_(const FrameCtx& ctx, const std::vector<Box>& tracks);

        MJPEGServer& server_;
        std::vector<IngestConfig> streams_;
        Options opt_;

        std::atomic<bool> running_{false};

        // global detector pool input
        BoundedQueue<FramePtr> infer_in_;
        BoundedQueue<TrackerFrameOutput> analytics_out_;

        std::vector<std::unique_ptr<StreamPipe>> pipes_;
        std::unordered_map<std::string, StreamPipe*> pipes_by_stream_id_;
        std::vector<std::thread> infer_pool_;

        std::shared_ptr<YuNetDetector> detector_;
        std::unique_ptr<Anonymizer> anonymizer_;
    };
}
