#include <pipeline/runtime.hpp>

#include <algorithm>
#include <chrono>
#include <map>
#include <iostream>
#include <utility>

#include "ingest/dual_source_factory.hpp"
#include "inference/yunet_detector.hpp"
#include "tracking/tracker.hpp"

namespace ss {
    PipelineRuntime::PipelineRuntime(MJPEGServer& server,
                                     std::vector<IngestConfig> streams,
                                     Options opt)
                                         : server_(server),
                                           streams_(std::move(streams)),
                                           opt_(opt),
                                           infer_in_(opt.infer_in_cap),
                                           analytics_out_(opt.analytics_cap) {}

    bool PipelineRuntime::start() {
        if (running_) return true;
        running_ = true;

        try {
            YuNetDetectorConfig cfg;
            cfg.param_path = opt_.detector_param_path;
            cfg.bin_path = opt_.detector_bin_path;
            cfg.input_w = opt_.detector_input_w;
            cfg.input_h = opt_.detector_input_h;
            cfg.score_threshold = opt_.detector_score_thresh;
            cfg.nms_threshold = opt_.detector_nms_thresh;
            cfg.top_k = opt_.detector_top_k;
            cfg.ncnn_threads = opt_.detector_ncnn_threads;
            detector_ = std::make_shared<YuNetDetector>(std::move(cfg));

            AnonymizerConfig acfg;
            acfg.method = opt_.anonymizer_method;
            acfg.pixelation_divisor = opt_.anonymizer_pixelation_divisor;
            acfg.blur_kernel = opt_.anonymizer_blur_kernel;
            anonymizer_ = std::make_unique<Anonymizer>(std::move(acfg));
        } catch (const std::exception& e) {
            std::cerr << "[Pipeline](start) module init failed: " << e.what() << "\n";
            running_ = false;
            return false;
        }

        pipes_.clear();
        pipes_by_stream_id_.clear();
        pipes_.reserve(streams_.size());

        for (const auto& s : streams_) {
            auto p = std::make_unique<StreamPipe>(s.id,
                                                  opt_.anon_in_cap,
                                                  opt_.anon_in_cap,
                                                  opt_.res_cap,
                                                  opt_.res_cap,
                                                  opt_.enc_in_cap);
            pipes_by_stream_id_[s.id] = p.get();
            pipes_.push_back(std::move(p));
        }

        infer_pool_.clear();
        infer_pool_.reserve(std::max(1, opt_.inf_workers));
        for (int i = 0; i < std::max(1, opt_.inf_workers); ++i) {
            infer_pool_.emplace_back([this]{ infer_loop_(); });
        }

        size_t started_streams = 0;
        for (size_t i = 0; i < streams_.size(); ++i) {
            const auto& cfg = streams_[i];
            StreamPipe* pipe = pipes_[i].get();

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

            pipe->inf_state_thr = std::thread([this, pipe] { infer_state_loop_(pipe); });
            pipe->anon_thr = std::thread([this, pipe] { anonymizer_loop_(pipe); });
            pipe->enc_thr = std::thread([this, pipe] { encoder_loop_(pipe); });
            ++started_streams;
        }

        if (started_streams == 0) {
            std::cerr << "[Pipeline](start) no streams were started.\n";
            stop();
            return false;
        }
        return true;
    }

    void PipelineRuntime::stop() {
        if (!running_) return;
        running_ = false;

        infer_in_.stop();
        analytics_out_.stop();
        for (auto& p : pipes_) {
            p->inf_state_in.stop();
            p->anon_in.stop();
            p->det_res.stop();
            p->inf_res.stop();
            p->enc_in.stop();
        }

        for (auto& p : pipes_) {
            if (p->ingest_thr.joinable()) p->ingest_thr.join();
            if (p->inf_state_thr.joinable()) p->inf_state_thr.join();
            if (p->anon_thr.joinable()) p->anon_thr.join();
            if (p->enc_thr.joinable()) p->enc_thr.join();
        }

        for (auto& t : infer_pool_) {
            if (t.joinable()) t.join();
        }
        infer_pool_.clear();
        detector_.reset();
        anonymizer_.reset();
    }

    bool PipelineRuntime::pop_tracker_output(TrackerFrameOutput& out, std::chrono::milliseconds timeout) {
        return analytics_out_.pop_for(out, timeout);
    }

    // loops)
    void PipelineRuntime::ingest_loop_(const IngestConfig& cfg,
                                       std::unique_ptr<GstDualSource> src,
                                       StreamPipe *pipe) {
        if (!src || !pipe) return;

        if (!src->start()) {
            std::cerr << "[Pipeline](ingest_loop_) start() failed for " << cfg.id << ".\n";
            return;
        }

        DualFramePacket dp;
        while (running_.load(std::memory_order_relaxed)) {
            if (!src->read(dp, 100)) continue;

            auto ctx = std::make_shared<FrameCtx>();
            ctx->stream_id = cfg.id;
            ctx->frame_id = dp.frame_id;
            ctx->pts_ns = dp.pts_ns;
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

            // branch
            infer_in_.push_drop_oldest(ctx); // global pool
            pipe->inf_state_in.push_drop_oldest(ctx); // per-stream tracker ordering path
            pipe->anon_in.push_drop_oldest(ctx); // per stream

        }
        src->stop();
    }

    void PipelineRuntime::infer_loop_() {
        while (running_.load(std::memory_order_relaxed)) {
            FramePtr ctx;
            if (!infer_in_.pop_for(ctx, std::chrono::milliseconds(200))) continue;
            if (!ctx) continue;

            InferResults res;
            res.stream_id = ctx->stream_id;
            res.frame_id = ctx->frame_id;

            try {
                res.bboxes = run_inference_(ctx->inf);
            } catch (const std::exception& e) {
                thread_local bool logged = false;
                if (!logged) {
                    std::cerr << "[Pipeline](infer_loop_) detector init failed: " << e.what() << "\n";
                    logged = true;
                }
                res.bboxes.clear();
            }
            // const std::string inf_key = ctx->stream_id + "/inf";
            // if (!ctx->inf.empty()) {
            //     server_.push_jpeg(inf_key, ctx->inf, opt_.jpeg_quality);
            //
            //     std::string inf_meta =
            //         "{"
            //         "\"stream_id\":\"" + ctx->stream_id + "\","
            //         "\"profile\":\"inf\","
            //         "\"frame_id\":" + std::to_string(ctx->frame_id) + ","
            //         "\"pts_ns\":" + std::to_string(ctx->pts_ns) + ","
            //         "\"w\":" + std::to_string(ctx->inf.cols) + ","
            //         "\"h\":" + std::to_string(ctx->inf.rows) +
            //         "}";
            //     server_.push_meta(inf_key, std::move(inf_meta));
            // }

            ctx->inf.release();

            auto it = pipes_by_stream_id_.find(ctx->stream_id);
            if (it != pipes_by_stream_id_.end() && it->second) {
                it->second->det_res.push_drop_oldest(std::move(res));
            }
        }
    }

    void PipelineRuntime::infer_state_loop_(StreamPipe* pipe) {
        if (!pipe) return;

        auto tracker = create_demo_tracker(TrackerConfig{});

        std::map<int64_t, FramePtr> pending_frames;
        std::map<int64_t, InferResults> pending_det_results;
        int64_t next_frame_id = -1;
        static constexpr int64_t kReorderWindow = 5;

        auto process_frame = [this, pipe, &tracker](const FramePtr& ctx, const std::vector<Box>& detections) {
            if (!ctx) return;

            InferResults tracked;
            tracked.stream_id = ctx->stream_id;
            tracked.frame_id = ctx->frame_id;
            tracked.bboxes = tracker->update(detections);

            publish_tracker_output_(*ctx, tracked.bboxes);
            pipe->inf_res.push_drop_oldest(std::move(tracked));
        };

        while (running_.load(std::memory_order_relaxed)) {
            FramePtr ctx;
            if (pipe->inf_state_in.pop_for(ctx, std::chrono::milliseconds(2)) && ctx) {
                pending_frames[ctx->frame_id] = ctx;
            }
            while (pipe->inf_state_in.try_pop(ctx)) {
                if (!ctx) continue;
                pending_frames[ctx->frame_id] = ctx;
            }

            InferResults det;
            while (pipe->det_res.try_pop(det)) {
                pending_det_results[det.frame_id] = std::move(det);
            }

            if (next_frame_id < 0 && !pending_frames.empty()) {
                next_frame_id = pending_frames.begin()->first;
            }

            while (next_frame_id >= 0) {
                auto fit = pending_frames.find(next_frame_id);
                auto dit = pending_det_results.find(next_frame_id);

                if (fit != pending_frames.end() && dit != pending_det_results.end()) {
                    process_frame(fit->second, dit->second.bboxes);
                    pending_frames.erase(fit);
                    pending_det_results.erase(dit);
                    ++next_frame_id;
                    continue;
                }

                if (fit == pending_frames.end()) {
                    if (!pending_frames.empty() && pending_frames.begin()->first > next_frame_id) {
                        next_frame_id = pending_frames.begin()->first;
                        continue;
                    }
                    break;
                }

                const int64_t latest_frame = pending_frames.empty() ? next_frame_id : pending_frames.rbegin()->first;
                const int64_t latest_det = pending_det_results.empty() ? next_frame_id : pending_det_results.rbegin()->first;
                const int64_t latest_seen = std::max(latest_frame, latest_det);
                if (latest_seen - next_frame_id > kReorderWindow) {
                    // Missing detector output for this frame: keep tracker state moving via predict-only step.
                    process_frame(fit->second, {});
                    pending_frames.erase(fit);
                    ++next_frame_id;
                    continue;
                }
                break;
            }

            while (pending_frames.size() > 500) pending_frames.erase(pending_frames.begin());
            while (pending_det_results.size() > 500) pending_det_results.erase(pending_det_results.begin());
        }
    }

    void PipelineRuntime::anonymizer_loop_(StreamPipe* pipe) {
        if (!pipe) return;

        std::map<int64_t, FramePtr> pending_frames;
        std::map<int64_t, InferResults> pending_track_results;

        while (running_.load(std::memory_order_relaxed)) {
            FramePtr ctx;
            if (pipe->anon_in.pop_for(ctx, std::chrono::milliseconds(2)) && ctx) {
                pending_frames[ctx->frame_id] = ctx;
            }

            while (pipe->anon_in.try_pop(ctx)) {
                if (!ctx) continue;
                pending_frames[ctx->frame_id] = ctx;
            }

            InferResults r;
            while (pipe->inf_res.try_pop(r)) {
                // `inf_res` carries tracker outputs produced by the inference state worker.
                pending_track_results[r.frame_id] = std::move(r);
            }

            for (auto fit = pending_frames.begin(); fit != pending_frames.end();) {
                auto rit = pending_track_results.find(fit->first);
                if (rit == pending_track_results.end()) {
                    ++fit;
                    continue;
                }

                FramePtr frame = fit->second;
                frame->tracked_boxes = rit->second.bboxes;
                anonymize_(frame->ui,
                           frame->tracked_boxes,
                           frame->scale_x,
                           frame->scale_y,
                           frame->offset_x,
                           frame->offset_y);
                pipe->enc_in.push_drop_oldest(frame);

                pending_track_results.erase(rit);
                fit = pending_frames.erase(fit);
            }

            while (pending_frames.size() > 500) pending_frames.erase(pending_frames.begin());
            while (pending_track_results.size() > 500) pending_track_results.erase(pending_track_results.begin());
        }
    }

    void PipelineRuntime::encoder_loop_(StreamPipe* pipe) {
        if (!pipe) return;

        const std::string ui_key = pipe->stream_id + "/ui";
        const std::string inf_key = pipe->stream_id + "/inf";

        while (running_.load(std::memory_order_relaxed)) {
            FramePtr ctx;
            if (!pipe->enc_in.pop_for(ctx, std::chrono::milliseconds(200))) continue;
            if (!ctx || ctx->ui.empty()) continue;

            server_.push_jpeg(ui_key, ctx->ui, opt_.jpeg_quality);

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
            server_.push_meta(ui_key, std::move(ui_meta));

            std::string inf_meta =
                "{"
                "\"stream_id\":\"" + ctx->stream_id + "\","
                "\"profile\":\"inf\","
                "\"frame_id\":" + std::to_string(ctx->frame_id) + ","
                "\"pts_ns\":" + std::to_string(ctx->pts_ns) + ","
                "\"w\":" + std::to_string(ctx->inf_w) + ","
                "\"h\":" + std::to_string(ctx->inf_h) +
                "}";
            server_.push_meta(inf_key, std::move(inf_meta));
        }
    }

    // hooks

    std::vector<Box> PipelineRuntime::run_inference_(const cv::Mat& inf) {
        if (!detector_) return {};
        return detector_->detect(inf);
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

    void PipelineRuntime::publish_tracker_output_(const FrameCtx& ctx, const std::vector<Box>& tracks) {
        TrackerFrameOutput out;
        out.stream_id = ctx.stream_id;
        out.frame_id = ctx.frame_id;
        out.pts_ns = ctx.pts_ns;
        out.tracks = tracks;
        analytics_out_.push_drop_oldest(std::move(out));
    }
}
