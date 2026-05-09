#include <pipeline/runtime.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "ingest/dual_source_factory.hpp"

namespace veilsight {
    namespace {
        uint64_t steady_now_ns() {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
        }

        int identity_worker_count(const IdentityModuleConfig& cfg) {
            return std::max(1, cfg.workers);
        }

        bool face_pipeline_enabled(const FaceDetectorModuleConfig& cfg) {
            return cfg.type != "none";
        }

        int recognizer_worker_count(const RecognizerModuleConfig& cfg) {
            return std::max(1, cfg.workers);
        }

        Box map_box_to_ui(const Box& box, const FrameCtx& frame) {
            Box out = box;
            out.x = box.x * frame.scale_x + frame.offset_x;
            out.y = box.y * frame.scale_y + frame.offset_y;
            out.w = box.w * frame.scale_x;
            out.h = box.h * frame.scale_y;
            if (out.face) {
                out.face->bbox.x = box.face->bbox.x * frame.scale_x + frame.offset_x;
                out.face->bbox.y = box.face->bbox.y * frame.scale_y + frame.offset_y;
                out.face->bbox.w = box.face->bbox.w * frame.scale_x;
                out.face->bbox.h = box.face->bbox.h * frame.scale_y;
                for (int i = 0; i < out.face->landmark_count && i < 5; ++i) {
                    out.face->landmarks[static_cast<size_t>(i)].x =
                        box.face->landmarks[static_cast<size_t>(i)].x * frame.scale_x + frame.offset_x;
                    out.face->landmarks[static_cast<size_t>(i)].y =
                        box.face->landmarks[static_cast<size_t>(i)].y * frame.scale_y + frame.offset_y;
                }
            }
            return out;
        }

        void add_queue_snapshot(std::map<std::string, QueueSnapshot>& out, const QueueSnapshot& snapshot, const std::string& name) {
            out[name] = snapshot;
        }
    }

    PipelineRuntime::StreamPipe::StreamPipe(std::string id,
                                            size_t frames_cap,
                                            size_t person_detections_cap,
                                            size_t faces_cap,
                                            size_t recognitions_cap,
                                            size_t identities_cap,
                                            size_t encoder_cap)
        : stream_id(std::move(id)),
          frames_in("stream/" + stream_id + "/frames.in",
                    "Ingestor",
                    "StreamCoordinator",
                    "Original FrameCtx with UI and inference mats.",
                    frames_cap),
          person_detections_in("stream/" + stream_id + "/person_detections.in",
                               "PersonDetectorPool",
                               "StreamCoordinator",
                               "Person detector results keyed by frame_id.",
                               person_detections_cap),
          faces_in("stream/" + stream_id + "/faces.in",
                   "FaceDetectorPool",
                   "StreamCoordinator",
                   "Face probe results keyed by frame_id and probe_id.",
                   faces_cap),
          recognitions_in("stream/" + stream_id + "/recognitions.in",
                          "RecognizerPool",
                          "StreamCoordinator",
                          "Recognition results keyed by frame_id.",
                          recognitions_cap),
          identities_in("stream/" + stream_id + "/identities.in",
                        "IdentityDecisionPool",
                        "StreamCoordinator",
                        "Track privacy decisions. Currently all unknown/anonymize.",
                        identities_cap),
          encoder_in("stream/" + stream_id + "/encoder.in",
                     "AnonymizerPool",
                     "Encoder",
                     "Final anonymized UI frame for publishing.",
                     encoder_cap) {}

    struct PipelineRuntime::PersonDetectorStage {
        explicit PersonDetectorStage(size_t input_cap, std::unique_ptr<IPersonDetectorFactory> stage_factory)
            : input("global/person_detector.in",
                    "Ingestor",
                    "PersonDetectorPool",
                    "Shared people detector work from all streams.",
                    input_cap),
              factory(std::move(stage_factory)) {}

        NamedQueue<PersonDetectionTask> input;
        std::unique_ptr<IPersonDetectorFactory> factory;
        std::vector<std::thread> workers;
    };

    struct PipelineRuntime::FaceDetectorStage {
        explicit FaceDetectorStage(size_t input_cap, std::unique_ptr<IFaceDetectorFactory> stage_factory)
            : input("global/face_detector.in",
                    "StreamCoordinator",
                    "FaceDetectorPool",
                    "Full-frame or ROI face probes from all streams.",
                    input_cap),
              factory(std::move(stage_factory)) {}

        NamedQueue<FaceDetectionTask> input;
        std::unique_ptr<IFaceDetectorFactory> factory;
        std::vector<std::thread> workers;
    };

    struct PipelineRuntime::IdentityStage {
        explicit IdentityStage(size_t input_cap, std::unique_ptr<IIdentityDeciderFactory> stage_factory)
            : input("global/identity.in",
                    "StreamCoordinator",
                    "IdentityDecisionPool",
                    "Future gallery/embedding work; noop/fail-closed for now.",
                    input_cap),
              factory(std::move(stage_factory)) {}

        NamedQueue<IdentityTask> input;
        std::unique_ptr<IIdentityDeciderFactory> factory;
        std::vector<std::thread> workers;
    };

    struct PipelineRuntime::RecognizerStage {
        explicit RecognizerStage(size_t input_cap, std::unique_ptr<IRecognizerFactory> stage_factory)
            : input("global/recognizer.in",
                    "StreamCoordinator",
                    "RecognizerPool",
                    "Embedding/face recognition work from all streams.",
                    input_cap),
              factory(std::move(stage_factory)) {}

        NamedQueue<RecognitionTask> input;
        std::unique_ptr<IRecognizerFactory> factory;
        std::vector<std::thread> workers;
    };

    PipelineRuntime::PipelineRuntime(IStreamPublisher& stream_publisher,
                                     ITelemetryPublisher& telemetry_publisher,
                                     std::vector<IngestConfig> streams,
                                     Options opt)
        : stream_publisher_(stream_publisher),
          telemetry_publisher_(telemetry_publisher),
          streams_(std::move(streams)),
          opt_(opt),
          anonymizer_in_("global/anonymizer.in",
                         "StreamCoordinator",
                         "AnonymizerPool",
                         "Shared anonymization work over UI frames.",
                         opt.anonymizer_in_cap) {}

    PipelineRuntime::~PipelineRuntime() {
        stop();
    }

    bool PipelineRuntime::start() {
        if (running_) return true;

        std::unique_ptr<IPersonDetectorFactory> detector_factory;
        std::unique_ptr<ITrackerFactory> tracker_factory;
        std::unique_ptr<IFaceDetectorFactory> face_detector_factory;
        std::unique_ptr<IRecognizerFactory> recognizer_factory;
        std::unique_ptr<IIdentityDeciderFactory> identity_factory;
        try {
            detector_factory = create_person_detector_factory(opt_.person_detector);
            tracker_factory = create_tracker_factory(opt_.tracker);
            if (face_pipeline_enabled(opt_.face_detector)) {
                face_detector_factory = create_face_detector_factory(opt_.face_detector);
            }
            recognizer_factory = create_recognizer_factory(opt_.recognizer);
            identity_factory = create_identity_decider_factory(opt_.identity);
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline](start) failed to create stage factories: " << e.what() << "\n";
            return false;
        }
        if (!detector_factory || !tracker_factory || !recognizer_factory || !identity_factory) {
            std::cerr << "[Pipeline](start) failed to create stage factories.\n";
            return false;
        }
        warn_if_oversubscribed_(*detector_factory, face_detector_factory.get(), *recognizer_factory, *identity_factory);

        anonymizer_in_.reset();
        if (opt_.metrics.enabled) {
            metrics_ = std::make_unique<RuntimeMetrics>();
        } else {
            metrics_.reset();
        }

        try {
            AnonymizerConfig acfg;
            acfg.method = opt_.anonymizer_method;
            acfg.pixelation_divisor = opt_.anonymizer_pixelation_divisor;
            acfg.blur_kernel = opt_.anonymizer_blur_kernel;
            anonymizer_ = std::make_unique<Anonymizer>(std::move(acfg));
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline](start) anonymizer init failed: " << e.what() << "\n";
            anonymizer_.reset();
            metrics_.reset();
            return false;
        }

        pipes_.clear();
        pipes_by_stream_id_.clear();
        pipes_.reserve(streams_.size());
        const bool face_enabled = face_pipeline_enabled(opt_.face_detector) && face_detector_factory != nullptr;
        for (const auto& s : streams_) {
            auto pipe = std::make_unique<StreamPipe>(s.id,
                                                     opt_.frames_in_cap,
                                                     opt_.person_detections_in_cap,
                                                     opt_.faces_in_cap,
                                                     opt_.recognitions_in_cap,
                                                     opt_.identities_in_cap,
                                                     opt_.encoder_in_cap);
            try {
                pipe->coordinator = std::make_unique<StreamCoordinator>(
                    tracker_factory->create(),
                    opt_.face_policy,
                    face_enabled,
                    opt_.reorder_window,
                    opt_.pending_state_limit);
            } catch (const std::exception& e) {
                std::cerr << "[Pipeline](start) stream coordinator init failed for " << s.id << ": " << e.what() << "\n";
                continue;
            }
            pipes_by_stream_id_[s.id] = pipe.get();
            pipes_.push_back(std::move(pipe));
        }

        person_detector_stage_ = std::make_unique<PersonDetectorStage>(opt_.person_detector_in_cap, std::move(detector_factory));
        face_detector_stage_ = std::make_unique<FaceDetectorStage>(opt_.face_detector_in_cap, std::move(face_detector_factory));
        recognizer_stage_ = std::make_unique<RecognizerStage>(opt_.recognizer_in_cap, std::move(recognizer_factory));
        identity_stage_ = std::make_unique<IdentityStage>(opt_.identity_in_cap, std::move(identity_factory));

        running_ = true;
        try {
            person_detector_stage_->workers.clear();
            person_detector_stage_->workers.reserve(std::max(1, opt_.person_detector.workers));
            for (int i = 0; i < std::max(1, opt_.person_detector.workers); ++i) {
                auto detector = person_detector_stage_->factory->create();
                person_detector_stage_->workers.emplace_back([this, detector = std::move(detector)]() mutable {
                    while (running_.load(std::memory_order_relaxed)) {
                        PersonDetectionTask task;
                        if (!person_detector_stage_->input.pop_for(task, std::chrono::milliseconds(200))) continue;
                        if (!detector) continue;

                        PersonDetectionResult result;
                        result.stream_id = task.stream_id;
                        result.frame_id = task.frame_id;
                        bool ok = true;
                        const uint64_t t0_ns = steady_now_ns();

                        try {
                            const auto boxes = detector->detect(task.input.image);
                            result.boxes.reserve(boxes.size());
                            for (const auto& box : boxes) {
                                result.boxes.push_back(map_box(task.input.image_to_frame, box));
                            }
                        } catch (const std::exception& e) {
                            ok = false;
                            thread_local bool logged = false;
                            if (!logged) {
                                std::cerr << "[Pipeline](person_detector) detect failed: " << e.what() << "\n";
                                logged = true;
                            }
                            result.boxes.clear();
                        }

                        if (metrics_) {
                            const uint64_t dt_ns = steady_now_ns() - t0_ns;
                            metrics_->observe_global(RuntimeStage::PersonDetector, dt_ns, ok);
                            metrics_->observe_stream(task.stream_id, RuntimeStage::PersonDetector, dt_ns, ok);
                        }

                        auto it = pipes_by_stream_id_.find(task.stream_id);
                        if (it != pipes_by_stream_id_.end() && it->second) {
                            it->second->person_detections_in.push_drop_oldest(std::move(result));
                        }
                    }
                });
            }

            if (face_detector_stage_->factory) {
                face_detector_stage_->workers.clear();
                face_detector_stage_->workers.reserve(std::max(1, opt_.face_detector.workers));
                for (int i = 0; i < std::max(1, opt_.face_detector.workers); ++i) {
                    auto detector = face_detector_stage_->factory->create();
                    face_detector_stage_->workers.emplace_back([this, detector = std::move(detector)]() mutable {
                        while (running_.load(std::memory_order_relaxed)) {
                            FaceDetectionTask task;
                            if (!face_detector_stage_->input.pop_for(task, std::chrono::milliseconds(200))) continue;
                            if (!detector) continue;

                            FaceDetectionResult result;
                            result.stream_id = task.stream_id;
                            result.frame_id = task.frame_id;
                            result.probe_id = task.probe_id;
                            result.kind = task.kind;
                            result.track_index = task.track_index;
                            result.roi = task.roi;
                            bool ok = true;
                            const uint64_t t0_ns = steady_now_ns();

                            try {
                                const auto faces = detector->detect_faces(task.input.image, task.run);
                                result.faces.reserve(faces.size());
                                for (const auto& face : faces) {
                                    result.faces.push_back(map_face(task.input.image_to_frame, face));
                                }
                            } catch (const std::exception& e) {
                                ok = false;
                                thread_local bool logged = false;
                                if (!logged) {
                                    std::cerr << "[Pipeline](face_detector) detect failed: " << e.what() << "\n";
                                    logged = true;
                                }
                                result.faces.clear();
                            }

                            if (metrics_) {
                                const uint64_t dt_ns = steady_now_ns() - t0_ns;
                                metrics_->observe_global(RuntimeStage::FaceDetector, dt_ns, ok);
                                metrics_->observe_stream(task.stream_id, RuntimeStage::FaceDetector, dt_ns, ok);
                            }

                            auto it = pipes_by_stream_id_.find(task.stream_id);
                            if (it != pipes_by_stream_id_.end() && it->second) {
                                it->second->faces_in.push_drop_oldest(std::move(result));
                            }
                        }
                    });
                }
            }

            recognizer_stage_->workers.clear();
            const int recognizer_workers = recognizer_worker_count(opt_.recognizer);
            recognizer_stage_->workers.reserve(recognizer_workers);
            for (int i = 0; i < recognizer_workers; ++i) {
                auto recognizer = recognizer_stage_->factory->create();
                recognizer_stage_->workers.emplace_back([this, recognizer = std::move(recognizer)]() mutable {
                    while (running_.load(std::memory_order_relaxed)) {
                        RecognitionTask task;
                        if (!recognizer_stage_->input.pop_for(task, std::chrono::milliseconds(200))) continue;
                        if (!recognizer) continue;

                        bool ok = true;
                        RecognitionResult result;
                        const uint64_t t0_ns = steady_now_ns();
                        try {
                            result = recognizer->recognize(task);
                        } catch (const std::exception& e) {
                            ok = false;
                            thread_local bool logged = false;
                            if (!logged) {
                                std::cerr << "[Pipeline](recognizer) recognize failed: " << e.what() << "\n";
                                logged = true;
                            }
                            result.stream_id = task.stream_id;
                            result.frame_id = task.frame_id;
                            result.frame = task.frame;
                            result.tracks = task.tracks;
                            for (auto& track : result.tracks) {
                                track.recognition_state = "failed";
                            }
                        }

                        if (metrics_) {
                            const uint64_t dt_ns = steady_now_ns() - t0_ns;
                            metrics_->observe_global(RuntimeStage::Recognizer, dt_ns, ok);
                            metrics_->observe_stream(task.stream_id, RuntimeStage::Recognizer, dt_ns, ok);
                        }

                        auto it = pipes_by_stream_id_.find(result.stream_id);
                        if (it != pipes_by_stream_id_.end() && it->second) {
                            it->second->recognitions_in.push_drop_oldest(std::move(result));
                        }
                    }
                });
            }

            identity_stage_->workers.clear();
            const int identity_workers = identity_worker_count(opt_.identity);
            identity_stage_->workers.reserve(identity_workers);
            for (int i = 0; i < identity_workers; ++i) {
                auto decider = identity_stage_->factory->create();
                identity_stage_->workers.emplace_back([this, decider = std::move(decider)]() mutable {
                    while (running_.load(std::memory_order_relaxed)) {
                        IdentityTask task;
                        if (!identity_stage_->input.pop_for(task, std::chrono::milliseconds(200))) continue;
                        if (!decider) continue;

                        bool ok = true;
                        IdentityResult result;
                        const uint64_t t0_ns = steady_now_ns();
                        try {
                            result = decider->decide(task);
                        } catch (const std::exception& e) {
                            ok = false;
                            thread_local bool logged = false;
                            if (!logged) {
                                std::cerr << "[Pipeline](identity) decide failed: " << e.what() << "\n";
                                logged = true;
                            }
                            result.stream_id = task.stream_id;
                            result.frame_id = task.frame_id;
                            result.frame = task.frame;
                            result.tracks = task.tracks;
                            for (auto& track : result.tracks) {
                                track.identity_key.clear();
                                track.identity_confidence = 0.0f;
                                track.privacy_action = "anonymize";
                            }
                        }

                        if (metrics_) {
                            const uint64_t dt_ns = steady_now_ns() - t0_ns;
                            metrics_->observe_global(RuntimeStage::Identity, dt_ns, ok);
                            metrics_->observe_stream(task.stream_id, RuntimeStage::Identity, dt_ns, ok);
                        }
                        publish_identity_result_(std::move(result));
                    }
                });
            }
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline](start) stage worker init failed: " << e.what() << "\n";
            stop();
            return false;
        }

        anonymizer_pool_.clear();
        anonymizer_pool_.reserve(std::max(1, opt_.anonymizer_workers));
        for (int i = 0; i < std::max(1, opt_.anonymizer_workers); ++i) {
            anonymizer_pool_.emplace_back([this] { anonymizer_loop_(); });
        }

        size_t started_streams = 0;
        for (const auto& cfg : streams_) {
            auto it = pipes_by_stream_id_.find(cfg.id);
            if (it == pipes_by_stream_id_.end() || !it->second) continue;
            StreamPipe* pipe = it->second;

            std::unique_ptr<GstDualSource> src;
            try {
                src = make_dual_source(cfg);
            } catch (const std::exception& e) {
                std::cerr << "[Pipeline](start) make_dual_source failed for " << cfg.id << ": " << e.what() << "\n";
                continue;
            }

            pipe->ingest_thr = std::thread([this, cfg, pipe, src = std::move(src)]() mutable {
                ingest_loop_(cfg, std::move(src), pipe);
            });
            pipe->coordinator_thr = std::thread([this, pipe] { coordinator_loop_(pipe); });
            pipe->enc_thr = std::thread([this, pipe] { encoder_loop_(pipe); });
            ++started_streams;
        }

        if (started_streams == 0) {
            std::cerr << "[Pipeline](start) no streams were started.\n";
            stop();
            return false;
        }

        if (metrics_) {
            metrics_thr_ = std::thread([this] { metrics_loop_(); });
        }
        return true;
    }

    void PipelineRuntime::stop() {
        if (!running_ && !person_detector_stage_ && !face_detector_stage_ && !recognizer_stage_ && !identity_stage_ &&
            pipes_.empty()) {
            return;
        }
        running_ = false;

        anonymizer_in_.stop();
        if (person_detector_stage_) person_detector_stage_->input.stop();
        if (face_detector_stage_) face_detector_stage_->input.stop();
        if (recognizer_stage_) recognizer_stage_->input.stop();
        if (identity_stage_) identity_stage_->input.stop();
        for (auto& pipe : pipes_) {
            pipe->frames_in.stop();
            pipe->person_detections_in.stop();
            pipe->faces_in.stop();
            pipe->recognitions_in.stop();
            pipe->identities_in.stop();
            pipe->encoder_in.stop();
        }

        for (auto& pipe : pipes_) {
            if (pipe->ingest_thr.joinable()) pipe->ingest_thr.join();
            if (pipe->coordinator_thr.joinable()) pipe->coordinator_thr.join();
            if (pipe->enc_thr.joinable()) pipe->enc_thr.join();
        }

        if (person_detector_stage_) {
            for (auto& worker : person_detector_stage_->workers) {
                if (worker.joinable()) worker.join();
            }
        }
        if (face_detector_stage_) {
            for (auto& worker : face_detector_stage_->workers) {
                if (worker.joinable()) worker.join();
            }
        }
        if (recognizer_stage_) {
            for (auto& worker : recognizer_stage_->workers) {
                if (worker.joinable()) worker.join();
            }
        }
        if (identity_stage_) {
            for (auto& worker : identity_stage_->workers) {
                if (worker.joinable()) worker.join();
            }
        }
        for (auto& worker : anonymizer_pool_) {
            if (worker.joinable()) worker.join();
        }
        anonymizer_pool_.clear();
        if (metrics_thr_.joinable()) metrics_thr_.join();

        pipes_.clear();
        pipes_by_stream_id_.clear();
        person_detector_stage_.reset();
        face_detector_stage_.reset();
        recognizer_stage_.reset();
        identity_stage_.reset();
        anonymizer_.reset();
        metrics_.reset();
    }

    bool PipelineRuntime::is_running() const {
        return running_.load(std::memory_order_relaxed);
    }

    void PipelineRuntime::ingest_loop_(const IngestConfig& cfg,
                                       std::unique_ptr<GstDualSource> src,
                                       StreamPipe* pipe) {
        if (!src || !pipe || !person_detector_stage_) return;
        if (!src->start()) {
            std::cerr << "[Pipeline](ingest_loop_) start() failed for " << cfg.id << ".\n";
            return;
        }

        DualFramePacket dp;
        while (running_.load(std::memory_order_relaxed)) {
            if (!src->read(dp, 100)) continue;
            const uint64_t ingest_t0_ns = steady_now_ns();

            auto ctx = std::make_shared<FrameCtx>();
            ctx->stream_id = cfg.id;
            ctx->frame_id = dp.frame_id;
            ctx->pts_ns = dp.pts_ns;
            ctx->created_steady_ns = ingest_t0_ns;
            ctx->scale_x = dp.scale_x;
            ctx->scale_y = dp.scale_y;
            ctx->offset_x = dp.offset_x;
            ctx->offset_y = dp.offset_y;
            ctx->ui = std::move(dp.ui_frame);
            ctx->inf = std::move(dp.inf_frame);
            ctx->inf_w = ctx->inf.cols;
            ctx->inf_h = ctx->inf.rows;
            ctx->ui_w = ctx->ui.cols;
            ctx->ui_h = ctx->ui.rows;

            PersonDetectionTask person_task;
            person_task.stream_id = ctx->stream_id;
            person_task.frame_id = ctx->frame_id;
            person_task.input.image = ctx->inf;
            person_task.input.image_to_frame = identity_transform(ctx->inf.size());
            person_task.frame_ctx = ctx;
            person_detector_stage_->input.push_drop_oldest(std::move(person_task));
            pipe->frames_in.push_drop_oldest(ctx);

            if (metrics_) {
                const uint64_t dt_ns = steady_now_ns() - ingest_t0_ns;
                metrics_->observe_global(RuntimeStage::Ingest, dt_ns);
                metrics_->observe_stream(cfg.id, RuntimeStage::Ingest, dt_ns);
            }
        }
        src->stop();
    }

    void PipelineRuntime::coordinator_loop_(StreamPipe* pipe) {
        if (!pipe || !pipe->coordinator) return;

        StreamCoordinator::Callbacks callbacks;
        callbacks.on_tracker_timing = [this](const FrameCtx& frame, uint64_t duration_ns) {
            if (!metrics_) return;
            metrics_->observe_global(RuntimeStage::Tracker, duration_ns);
            metrics_->observe_stream(frame.stream_id, RuntimeStage::Tracker, duration_ns);
        };
        callbacks.on_face_probes_ready = [this](std::vector<FaceDetectionTask> probes) {
            if (!face_detector_stage_) return;
            for (auto& probe : probes) {
                face_detector_stage_->input.push_drop_oldest(std::move(probe));
            }
        };
        callbacks.on_identity_ready = [this](IdentityTask task) {
            if (!identity_stage_) return;
            identity_stage_->input.push_drop_oldest(std::move(task));
        };
        callbacks.on_recognition_ready = [this](RecognitionTask task) {
            if (!recognizer_stage_) return;
            recognizer_stage_->input.push_drop_oldest(std::move(task));
        };
        callbacks.on_frame_committed = [this](const FramePtr& frame) {
            commit_frame_(frame);
        };

        while (running_.load(std::memory_order_relaxed)) {
            FramePtr frame;
            if (pipe->frames_in.pop_for(frame, std::chrono::milliseconds(2)) && frame) {
                pipe->coordinator->push_frame(frame);
            }
            while (pipe->frames_in.try_pop(frame)) {
                if (frame) pipe->coordinator->push_frame(frame);
            }

            PersonDetectionResult person_result;
            while (pipe->person_detections_in.try_pop(person_result)) {
                pipe->coordinator->push_person_detection(std::move(person_result));
            }

            FaceDetectionResult face_result;
            while (pipe->faces_in.try_pop(face_result)) {
                pipe->coordinator->push_face_result(std::move(face_result));
            }

            RecognitionResult recognition_result;
            while (pipe->recognitions_in.try_pop(recognition_result)) {
                pipe->coordinator->push_recognition_result(std::move(recognition_result));
            }

            IdentityResult identity_result;
            while (pipe->identities_in.try_pop(identity_result)) {
                pipe->coordinator->push_identity_result(std::move(identity_result));
            }

            pipe->coordinator->drain_ready(callbacks);
        }
    }

    void PipelineRuntime::publish_identity_result_(IdentityResult result) {
        auto it = pipes_by_stream_id_.find(result.stream_id);
        if (it == pipes_by_stream_id_.end() || !it->second) return;
        it->second->identities_in.push_drop_oldest(std::move(result));
    }

    void PipelineRuntime::commit_frame_(const FramePtr& frame) {
        if (!frame) return;
        publish_frame_analytics_(*frame, frame->tracked_boxes);
        frame->inf.release();
        AnonymizeTask task;
        task.stream_id = frame->stream_id;
        task.frame_id = frame->frame_id;
        task.frame = frame;
        anonymizer_in_.push_drop_oldest(std::move(task));
    }

    void PipelineRuntime::anonymizer_loop_() {
        while (running_.load(std::memory_order_relaxed)) {
            AnonymizeTask task;
            if (!anonymizer_in_.pop_for(task, std::chrono::milliseconds(200))) continue;
            FramePtr ctx = task.frame;
            if (!ctx) continue;
            const uint64_t anonymizer_t0_ns = steady_now_ns();

            anonymize_(ctx->ui,
                       ctx->tracked_boxes,
                       ctx->scale_x,
                       ctx->scale_y,
                       ctx->offset_x,
                       ctx->offset_y);

            if (metrics_) {
                const uint64_t dt_ns = steady_now_ns() - anonymizer_t0_ns;
                metrics_->observe_global(RuntimeStage::Anonymizer, dt_ns);
                metrics_->observe_stream(ctx->stream_id, RuntimeStage::Anonymizer, dt_ns);
            }

            auto it = pipes_by_stream_id_.find(ctx->stream_id);
            if (it != pipes_by_stream_id_.end() && it->second) {
                AnonymizeResult result;
                result.stream_id = ctx->stream_id;
                result.frame_id = ctx->frame_id;
                result.frame = ctx;
                it->second->encoder_in.push_drop_oldest(std::move(result));
            }
        }
    }

    void PipelineRuntime::encoder_loop_(StreamPipe* pipe) {
        if (!pipe) return;
        const std::string ui_key = pipe->stream_id + "/ui";

        while (running_.load(std::memory_order_relaxed)) {
            AnonymizeResult result;
            if (!pipe->encoder_in.pop_for(result, std::chrono::milliseconds(200))) continue;
            FramePtr ctx = result.frame;
            if (!ctx || ctx->ui.empty()) continue;
            const uint64_t encoder_t0_ns = steady_now_ns();

            stream_publisher_.publish_frame(ui_key, ctx->ui, opt_.jpeg_quality);

            std::string ui_meta =
                "{"
                "\"stream_id\":\"" + ctx->stream_id + "\"," 
                "\"profile\":\"ui\"," 
                "\"frame_id\":" + std::to_string(ctx->frame_id) + ","
                "\"pts_ns\":" + std::to_string(ctx->pts_ns) + ","
                "\"w\":" + std::to_string(ctx->ui.cols) + ","
                "\"h\":" + std::to_string(ctx->ui.rows) + ","
                "\"tracks\":" + std::to_string(ctx->tracked_boxes.size()) +
                "}";
            stream_publisher_.publish_metadata(ui_key, std::move(ui_meta));

            if (metrics_) {
                const uint64_t encoder_t1_ns = steady_now_ns();
                const uint64_t encoder_dt_ns = encoder_t1_ns - encoder_t0_ns;
                metrics_->observe_global(RuntimeStage::Encoder, encoder_dt_ns);
                metrics_->observe_stream(ctx->stream_id, RuntimeStage::Encoder, encoder_dt_ns);
                if (ctx->created_steady_ns > 0 && encoder_t1_ns >= ctx->created_steady_ns) {
                    const uint64_t e2e_ns = encoder_t1_ns - ctx->created_steady_ns;
                    metrics_->observe_global(RuntimeStage::EndToEnd, e2e_ns);
                    metrics_->observe_stream(ctx->stream_id, RuntimeStage::EndToEnd, e2e_ns);
                }
            }
        }
    }

    void PipelineRuntime::warn_if_oversubscribed_(const IPersonDetectorFactory& detector_factory,
                                                  const IFaceDetectorFactory* face_detector_factory,
                                                  const IRecognizerFactory& recognizer_factory,
                                                  const IIdentityDeciderFactory& identity_factory) const {
        const size_t cpu_budget = static_cast<size_t>(std::max(1u, std::thread::hardware_concurrency()));
        const size_t stream_threads = streams_.size() * 3u;
        const size_t person_detector_parallelism =
            static_cast<size_t>(std::max(1, opt_.person_detector.workers)) *
            static_cast<size_t>(std::max(1, detector_factory.backend_threads()));
        const size_t face_detector_parallelism =
            face_detector_factory
                ? static_cast<size_t>(std::max(1, opt_.face_detector.workers)) *
                      static_cast<size_t>(std::max(1, face_detector_factory->backend_threads()))
                : 0u;
        const size_t recognizer_parallelism =
            static_cast<size_t>(recognizer_worker_count(opt_.recognizer)) *
            static_cast<size_t>(std::max(1, recognizer_factory.backend_threads()));
        const size_t identity_parallelism =
            static_cast<size_t>(identity_worker_count(opt_.identity)) *
            static_cast<size_t>(std::max(1, identity_factory.backend_threads()));
        const size_t anonymizer_threads = static_cast<size_t>(std::max(1, opt_.anonymizer_workers));
        const size_t metrics_threads = opt_.metrics.enabled ? 1u : 0u;
        const size_t total_parallelism =
            stream_threads + person_detector_parallelism + face_detector_parallelism +
            recognizer_parallelism + identity_parallelism + anonymizer_threads + metrics_threads;

        if (total_parallelism <= cpu_budget) {
            return;
        }

        std::cerr << "[Pipeline](start) warning: oversubscribed configuration: required_threads="
                  << total_parallelism
                  << " cpu_budget=" << cpu_budget
                  << " stream_threads=" << stream_threads
                  << " person_detector_parallelism=" << person_detector_parallelism
                  << " face_detector_parallelism=" << face_detector_parallelism
                  << " recognizer_parallelism=" << recognizer_parallelism
                  << " identity_parallelism=" << identity_parallelism
                  << " anonymizer_threads=" << anonymizer_threads
                  << " metrics_threads=" << metrics_threads
                  << "\n";
    }

    void PipelineRuntime::anonymize_(cv::Mat& ui,
                                     const std::vector<Box>& bboxes,
                                     float sx,
                                     float sy,
                                     float tx,
                                     float ty) {
        if (!anonymizer_) return;
        anonymizer_->apply(ui, bboxes, sx, sy, tx, ty);
    }

    void PipelineRuntime::draw_tracks_(cv::Mat& ui,
                                       const std::vector<Box>& boxes,
                                       float sx,
                                       float sy,
                                       float tx,
                                       float ty) {
        if (ui.empty()) return;

        for (const auto& b : boxes) {
            const int x = static_cast<int>(std::lround(b.x * sx + tx));
            const int y = static_cast<int>(std::lround(b.y * sy + ty));
            const int w = static_cast<int>(std::lround(b.w * sx));
            const int h = static_cast<int>(std::lround(b.h * sy));

            cv::Rect r(x, y, w, h);
            cv::Rect bounds(0, 0, ui.cols, ui.rows);
            r &= bounds;
            if (r.width < 2 || r.height < 2) continue;

            const cv::Scalar color = b.occluded ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(ui, r, color, 2);

            const std::string label = "id:" + std::to_string(b.id);
            const int text_y = std::max(14, r.y - 4);
            cv::putText(ui,
                        label,
                        cv::Point(r.x, text_y),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        1,
                        cv::LINE_AA);
        }
    }

    void PipelineRuntime::publish_frame_analytics_(const FrameCtx& ctx, const std::vector<Box>& tracks) {
        std::vector<Box> ui_tracks;
        ui_tracks.reserve(tracks.size());
        for (const auto& track : tracks) {
            ui_tracks.push_back(map_box_to_ui(track, ctx));
        }

        telemetry_publisher_.publish_frame_analytics(ctx, ui_tracks);
    }

    std::map<std::string, QueueSnapshot> PipelineRuntime::snapshot_queues_() const {
        std::map<std::string, QueueSnapshot> out;
        if (person_detector_stage_) {
            add_queue_snapshot(out, person_detector_stage_->input.snapshot(), person_detector_stage_->input.name());
        }
        if (face_detector_stage_) {
            add_queue_snapshot(out, face_detector_stage_->input.snapshot(), face_detector_stage_->input.name());
        }
        if (recognizer_stage_) {
            add_queue_snapshot(out, recognizer_stage_->input.snapshot(), recognizer_stage_->input.name());
        }
        if (identity_stage_) {
            add_queue_snapshot(out, identity_stage_->input.snapshot(), identity_stage_->input.name());
        }
        add_queue_snapshot(out, anonymizer_in_.snapshot(), anonymizer_in_.name());

        for (const auto& pipe : pipes_) {
            if (!pipe) continue;
            add_queue_snapshot(out, pipe->frames_in.snapshot(), pipe->frames_in.name());
            add_queue_snapshot(out, pipe->person_detections_in.snapshot(), pipe->person_detections_in.name());
            add_queue_snapshot(out, pipe->faces_in.snapshot(), pipe->faces_in.name());
            add_queue_snapshot(out, pipe->recognitions_in.snapshot(), pipe->recognitions_in.name());
            add_queue_snapshot(out, pipe->identities_in.snapshot(), pipe->identities_in.name());
            add_queue_snapshot(out, pipe->encoder_in.snapshot(), pipe->encoder_in.name());
        }
        return out;
    }

    void PipelineRuntime::metrics_loop_() {
        if (!metrics_) return;

        const auto tick = std::chrono::milliseconds(250);
        auto next_log = std::chrono::steady_clock::now();

        while (running_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(tick);
            if (!running_.load(std::memory_order_relaxed)) break;

            const RuntimeMetrics::Snapshot snap = metrics_->snapshot();
            const auto queues = snapshot_queues_();
            const std::string json = metrics_snapshot_to_json(snap, queues);

            if (opt_.metrics.enable_http || opt_.metrics.enable_ui_payload) {
                stream_publisher_.publish_metrics(json);
            }
            telemetry_publisher_.publish_metrics_snapshot(snap, queues);

            const auto now = std::chrono::steady_clock::now();
            if (now < next_log) continue;
            next_log = now + std::chrono::milliseconds(opt_.metrics.log_interval_ms);

            const auto pick_stage = [&snap](RuntimeStage stage) -> StageSnapshot {
                auto it = snap.global.find(stage);
                return (it != snap.global.end()) ? it->second : StageSnapshot{};
            };
            const StageSnapshot person = pick_stage(RuntimeStage::PersonDetector);
            const StageSnapshot trk = pick_stage(RuntimeStage::Tracker);
            const StageSnapshot face = pick_stage(RuntimeStage::FaceDetector);
            const StageSnapshot rec = pick_stage(RuntimeStage::Recognizer);
            const StageSnapshot identity = pick_stage(RuntimeStage::Identity);
            const StageSnapshot ano = pick_stage(RuntimeStage::Anonymizer);
            const StageSnapshot enc = pick_stage(RuntimeStage::Encoder);
            const StageSnapshot e2e = pick_stage(RuntimeStage::EndToEnd);

            std::cerr << "[Metrics] "
                      << "person_fps=" << person.fps << " person_p95_ms=" << person.p95_ms << " "
                      << "trk_fps=" << trk.fps << " trk_p95_ms=" << trk.p95_ms << " "
                      << "face_fps=" << face.fps << " face_p95_ms=" << face.p95_ms << " "
                      << "rec_fps=" << rec.fps << " rec_p95_ms=" << rec.p95_ms << " "
                      << "identity_fps=" << identity.fps << " identity_p95_ms=" << identity.p95_ms << " "
                      << "anon_fps=" << ano.fps << " anon_p95_ms=" << ano.p95_ms << " "
                      << "enc_fps=" << enc.fps << " enc_p95_ms=" << enc.p95_ms << " "
                      << "e2e_p95_ms=" << e2e.p95_ms
                      << "\n";

            for (const auto& [name, q] : queues) {
                if (q.capacity == 0) continue;
                if ((100 * q.size) / q.capacity >= 80) {
                    std::cerr << "[Metrics] queue_high_watermark " << name
                              << " size=" << q.size
                              << " cap=" << q.capacity
                              << " dropped=" << q.dropped
                              << "\n";
                }
            }
        }
    }
}
