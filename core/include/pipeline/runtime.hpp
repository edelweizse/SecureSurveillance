#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <anonymization/anonymizer.hpp>
#include <common/config.hpp>
#include <person_detector/person_detector.hpp>
#include <face_detector/face_detector.hpp>
#include <identity/identity_decider.hpp>
#include <ingest/gst_dual_source.hpp>
#include <pipeline/bounded_queue.hpp>
#include <pipeline/lifecycle.hpp>
#include <pipeline/metrics.hpp>
#include <pipeline/publishers.hpp>
#include <pipeline/stream_coordinator.hpp>
#include <pipeline/tasks.hpp>
#include <pipeline/types.hpp>
#include <recognizer/recognizer.hpp>

namespace veilsight {
    class PipelineRuntime : public IPipelineLifecycle {
    public:
        struct Options {
            int jpeg_quality = 75;

            size_t person_detector_in_cap = 50;
            size_t frames_in_cap = 5;
            size_t person_detections_in_cap = 20;
            size_t face_detector_in_cap = 50;
            size_t faces_in_cap = 20;
            size_t recognizer_in_cap = 50;
            size_t recognitions_in_cap = 20;
            size_t identity_in_cap = 50;
            size_t identities_in_cap = 20;
            size_t anonymizer_in_cap = 50;
            size_t encoder_in_cap = 5;

            int64_t reorder_window = 5;
            size_t pending_state_limit = 500;
            int anonymizer_workers = 1;

            PersonDetectorModuleConfig person_detector;
            TrackerModuleConfig tracker;
            FaceDetectorModuleConfig face_detector;
            RecognizerModuleConfig recognizer;
            IdentityModuleConfig identity;

            std::string anonymizer_method = "pixelate";
            int anonymizer_pixelation_divisor = 10;
            int anonymizer_blur_kernel = 31;
            bool anonymizer_face_only_when_available = false;

            MetricsConfig metrics;
        };

        PipelineRuntime(IStreamPublisher& stream_publisher,
                        ITelemetryPublisher& telemetry_publisher,
                        std::vector<IngestConfig> streams,
                        Options opt);

        bool start() override;
        void stop() override;
        bool is_running() const override;
        bool reload_recognizer_gallery(std::string* error);

        ~PipelineRuntime();

    private:
        struct StreamPipe {
            std::string stream_id;

            NamedQueue<FramePtr> frames_in;
            NamedQueue<PersonDetectionResult> person_detections_in;
            NamedQueue<FaceDetectionResult> faces_in;
            NamedQueue<RecognitionResult> recognitions_in;
            NamedQueue<IdentityResult> identities_in;
            NamedQueue<AnonymizeResult> encoder_in;
            std::unique_ptr<StreamCoordinator> coordinator;

            std::thread ingest_thr;
            std::thread coordinator_thr;
            std::thread enc_thr;

            StreamPipe(std::string id,
                       size_t frames_cap,
                       size_t person_detections_cap,
                       size_t faces_cap,
                       size_t recognitions_cap,
                       size_t identities_cap,
                       size_t encoder_cap);
        };

        struct PersonDetectorStage;
        struct FaceDetectorStage;
        struct RecognizerStage;
        struct IdentityStage;

        void ingest_loop_(const IngestConfig& cfg, std::unique_ptr<GstDualSource> src, StreamPipe* pipe);
        void coordinator_loop_(StreamPipe* pipe);
        void anonymizer_loop_();
        void encoder_loop_(StreamPipe* pipe);
        void metrics_loop_();

        void publish_identity_result_(IdentityResult result);
        void commit_frame_(const FramePtr& frame);
        void warn_if_oversubscribed_(const IPersonDetectorFactory& detector_factory,
                                     const IFaceDetectorFactory* face_detector_factory,
                                     const IRecognizerFactory& recognizer_factory,
                                     const IIdentityDeciderFactory& identity_factory) const;

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
        void publish_frame_analytics_(const FrameCtx& ctx, const std::vector<Box>& tracks);
        std::map<std::string, QueueSnapshot> snapshot_queues_() const;

        IStreamPublisher& stream_publisher_;
        ITelemetryPublisher& telemetry_publisher_;
        std::vector<IngestConfig> streams_;
        Options opt_;

        std::atomic<bool> running_{false};
        std::atomic<uint64_t> person_detections_total_{0};
        std::atomic<uint64_t> face_detections_total_{0};
        std::atomic<uint64_t> committed_tracks_total_{0};
        NamedQueue<AnonymizeTask> anonymizer_in_;

        std::vector<std::unique_ptr<StreamPipe>> pipes_;
        std::unordered_map<std::string, StreamPipe*> pipes_by_stream_id_;
        std::unique_ptr<PersonDetectorStage> person_detector_stage_;
        std::unique_ptr<FaceDetectorStage> face_detector_stage_;
        std::unique_ptr<RecognizerStage> recognizer_stage_;
        std::unique_ptr<IdentityStage> identity_stage_;
        std::vector<std::thread> anonymizer_pool_;
        std::thread metrics_thr_;

        std::unique_ptr<Anonymizer> anonymizer_;
        std::unique_ptr<RuntimeMetrics> metrics_;
    };
}
