#include "runner_manager.hpp"

#include <chrono>
#include <exception>
#include <utility>

#include <common/replicate.hpp>

#include <opencv2/imgcodecs.hpp>

namespace veilsight {
    namespace {
        uint64_t now_epoch_ms() {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count());
        }
    }

    RunnerManager::RunnerManager(std::string config_path, AppConfig config, TelemetryHub& telemetry)
        : config_path_(std::move(config_path)),
          config_(std::move(config)),
          expanded_streams_(expand_replicas(config_.streams)),
          http_server_(config_.server.url, config_.server.port, config_.streaming),
          mjpeg_publisher_(http_server_),
          webrtc_publisher_(config_.streaming),
          stream_publisher_(mjpeg_publisher_,
                            webrtc_publisher_,
                            config_.streaming.fallback == "mjpeg" || config_.streaming.primary == "mjpeg"),
          telemetry_(telemetry) {
        telemetry_.set_streams(stream_ids_(expanded_streams_));
        http_server_.set_webrtc_publisher(&webrtc_publisher_);
        http_server_.start();
        register_streams_locked_();
    }

    bool RunnerManager::start(std::string* error) {
        std::lock_guard lk(mutex_);
        return start_locked_(error);
    }

    void RunnerManager::stop() {
        std::lock_guard lk(mutex_);
        stop_locked_();
    }

    bool RunnerManager::reload_config_yaml(const std::string& config_yaml, std::string* error) {
        AppConfig new_config;
        std::vector<IngestConfig> new_streams;
        try {
            new_config = config_yaml.empty() ? load_config_yaml(config_path_)
                                             : load_config_yaml_string(config_yaml);
            new_streams = expand_replicas(new_config.streams);
            if (new_streams.empty()) {
                if (error) *error = "no streams configured";
                return false;
            }
        } catch (const std::exception& e) {
            if (error) *error = e.what();
            return false;
        }

        std::lock_guard lk(mutex_);
        AppConfig old_config = config_;
        std::vector<IngestConfig> old_streams = expanded_streams_;
        stop_locked_();
        config_ = std::move(new_config);
        expanded_streams_ = std::move(new_streams);
        telemetry_.set_streams(stream_ids_(expanded_streams_));
        register_streams_locked_();
        if (start_locked_(error)) {
            return true;
        }

        const std::string reload_error = error ? *error : "reload failed";
        last_error_ = reload_error;
        config_ = std::move(old_config);
        expanded_streams_ = std::move(old_streams);
        telemetry_.set_streams(stream_ids_(expanded_streams_));
        register_streams_locked_();
        std::string rollback_error;
        const bool restored = start_locked_(&rollback_error);
        if (error) {
            *error = restored ? reload_error
                              : reload_error + "; rollback failed: " + rollback_error;
        }
        return false;
    }

    bool RunnerManager::reload_gallery(std::string* error) {
        std::lock_guard lk(mutex_);
        if (!runtime_ || !runtime_->is_running()) {
            if (error) *error = "pipeline is not running";
            return false;
        }
        return runtime_->reload_recognizer_gallery(error);
    }

    EnrollmentAnalysisResult RunnerManager::analyze_enrollment_image(
        const std::string& image_bytes,
        const std::string& mime_type) const {
        EnrollmentAnalysisResult out;
        if (mime_type != "image/jpeg" && mime_type != "image/png" && mime_type != "image/webp") {
            out.ok = false;
            out.message = "unsupported image MIME type";
            return out;
        }
        std::vector<unsigned char> bytes(image_bytes.begin(), image_bytes.end());
        cv::Mat bgr = cv::imdecode(bytes, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            out.ok = false;
            out.message = "failed to decode image";
            return out;
        }

        FaceDetectorModuleConfig face_cfg;
        RecognizerModuleConfig recognizer_cfg;
        {
            std::lock_guard lk(mutex_);
            face_cfg = config_.modules.face_detector;
            recognizer_cfg = config_.modules.recognizer;
        }
        if (face_cfg.type == "none") {
            face_cfg.type = "scrfd";
        }
        if (recognizer_cfg.type != "mobilefacenet") {
            recognizer_cfg.type = "mobilefacenet";
        }
        return analyze_mobilefacenet_enrollment_image(face_cfg, recognizer_cfg, bgr);
    }

    bool RunnerManager::validate_config_yaml(const std::string& config_yaml,
                                             std::vector<std::string>* errors) const {
        try {
            const AppConfig parsed = config_yaml.empty() ? load_config_yaml(config_path_)
                                                        : load_config_yaml_string(config_yaml);
            const auto streams = expand_replicas(parsed.streams);
            if (streams.empty()) {
                if (errors) errors->push_back("no streams configured");
                return false;
            }
            return true;
        } catch (const std::exception& e) {
            if (errors) errors->push_back(e.what());
            return false;
        }
    }

    runner::v1::PipelineStatus RunnerManager::status(std::string message) const {
        std::lock_guard lk(mutex_);
        runner::v1::PipelineStatus out;
        out.set_runner_id(config_.runner.id);
        const bool running = runtime_ && runtime_->is_running();
        out.set_running(running);
        out.set_state(state_);
        if (out.state().empty()) out.set_state(running ? "running" : "stopped");
        out.set_message(message.empty() ? last_error_ : std::move(message));
        out.set_timestamp_ms(now_epoch_ms());
        for (const auto& stream : expanded_streams_) {
            auto* ref = out.add_streams();
            ref->set_stream_id(stream.id);
            ref->set_profile("ui");
        }
        return out;
    }

    runner::v1::StreamList RunnerManager::streams() const {
        std::lock_guard lk(mutex_);
        runner::v1::StreamList out;
        out.set_public_base_url(config_.runner.public_base_url);
        for (const auto& stream : expanded_streams_) {
            auto* ref = out.add_streams();
            ref->set_stream_id(stream.id);
            ref->set_profile("ui");
            ref->set_webrtc_available(webrtc_publisher_.available());
        }
        return out;
    }

    PipelineRuntime::Options RunnerManager::make_options_(const AppConfig& config) {
        PipelineRuntime::Options opt;
        opt.person_detector = config.modules.person_detector;
        opt.tracker = config.modules.tracker;
        opt.face_detector = config.modules.face_detector;
        opt.recognizer = config.modules.recognizer;
        opt.identity = config.modules.identity;
        opt.metrics = config.metrics;
        opt.person_detector_in_cap = config.runtime.queues.global.person_detector_in_capacity;
        opt.face_detector_in_cap = config.runtime.queues.global.face_detector_in_capacity;
        opt.recognizer_in_cap = config.runtime.queues.global.recognizer_in_capacity;
        opt.identity_in_cap = config.runtime.queues.global.identity_in_capacity;
        opt.anonymizer_in_cap = config.runtime.queues.global.anonymizer_in_capacity;
        opt.frames_in_cap = config.runtime.queues.per_stream.frames_in_capacity;
        opt.person_detections_in_cap = config.runtime.queues.per_stream.person_detections_in_capacity;
        opt.faces_in_cap = config.runtime.queues.per_stream.faces_in_capacity;
        opt.recognitions_in_cap = config.runtime.queues.per_stream.recognitions_in_capacity;
        opt.identities_in_cap = config.runtime.queues.per_stream.identities_in_capacity;
        opt.encoder_in_cap = config.runtime.queues.per_stream.encoder_in_capacity;
        opt.reorder_window = config.runtime.reorder_window;
        opt.pending_state_limit = config.runtime.pending_state_limit;
        opt.jpeg_quality = config.runtime.jpeg_quality;
        opt.anonymizer_workers = config.runtime.anonymizer.model_instances;
        opt.anonymizer_method = config.runtime.anonymizer.method;
        opt.anonymizer_pixelation_divisor = config.runtime.anonymizer.pixelation_divisor;
        opt.anonymizer_blur_kernel = config.runtime.anonymizer.blur_kernel;
        return opt;
    }

    std::vector<std::string> RunnerManager::stream_ids_(const std::vector<IngestConfig>& streams) {
        std::vector<std::string> out;
        out.reserve(streams.size());
        for (const auto& stream : streams) out.push_back(stream.id);
        return out;
    }

    bool RunnerManager::start_locked_(std::string* error) {
        if (runtime_ && runtime_->is_running()) return true;
        if (expanded_streams_.empty()) {
            if (error) *error = "no streams configured";
            last_error_ = error ? *error : "no streams configured";
            state_ = "error";
            return false;
        }

        state_ = "starting";
        runtime_ = std::make_unique<PipelineRuntime>(stream_publisher_,
                                                     telemetry_,
                                                     expanded_streams_,
                                                     make_options_(config_));
        if (!runtime_->start()) {
            runtime_.reset();
            if (error) *error = "pipeline failed to start";
            last_error_ = error ? *error : "pipeline failed to start";
            state_ = "error";
            telemetry_.publish_status("stopped", "pipeline failed to start");
            return false;
        }
        state_ = "running";
        last_error_.clear();
        telemetry_.publish_status("running", "pipeline started");
        return true;
    }

    void RunnerManager::stop_locked_() {
        if (!runtime_) return;
        state_ = "stopping";
        runtime_->stop();
        runtime_.reset();
        webrtc_publisher_.clear_sessions();
        state_ = "stopped";
        telemetry_.publish_status("stopped", "pipeline stopped");
    }

    void RunnerManager::register_streams_locked_() {
        for (const auto& stream : expanded_streams_) {
            stream_publisher_.register_stream(stream.id + "/ui");
        }
    }
}
