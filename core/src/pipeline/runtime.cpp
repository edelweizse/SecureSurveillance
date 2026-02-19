#include <pipeline/runtime.hpp>

#include <chrono>
#include <iostream>
#include <unordered_map>

#include "ingest/dual_source_factory.hpp"

namespace ss {
    PipelineRuntime::PipelineRuntime(MJPEGServer& server,
                                     std::vector<IngestConfig> streams,
                                     Options opt)
                                         : server_(server),
                                           streams_(std::move(streams)),
                                           opt_(opt),
                                           infer_in_(opt.infer_in_cap) {}

    bool PipelineRuntime::start() {
        if (running_) return true;
        running_ = true;

        pipes_.clear();
        pipes_by_stream_id_.clear();
        pipes_.reserve(streams_.size());

        for (const auto& s : streams_) {
            auto p = std::make_unique<StreamPipe>(s.id,
                                                  opt_.anon_in_cap,
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

            pipe->anon_thr = std::thread([this, pipe] { anonymizer_loop_(pipe); });
            pipe->enc_thr = std::thread([this, pipe] { encoder_loop_(pipe); });
        }
        return true;
    }

    void PipelineRuntime::stop() {
        if (!running_) return;
        running_ = false;

        infer_in_.stop();
        for (auto& p : pipes_) {
            p->anon_in.stop();
            p->inf_res.stop();
            p->enc_in.stop();
        }

        for (auto& p : pipes_) {
            if (p->ingest_thr.joinable()) p->ingest_thr.join();
            if (p->anon_thr.joinable()) p->anon_thr.join();
            if (p->enc_thr.joinable()) p->enc_thr.join();
        }

        for (auto& t : infer_pool_) {
            if (t.joinable()) t.join();
        }
        infer_pool_.clear();
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
            if (!src->read(dp, 100)) continue;;

            auto ctx = std::make_shared<FrameCtx>();
            ctx->stream_id = cfg.id;
            ctx->frame_id = dp.frame_id;
            ctx->pts_ns = dp.pts_ns;
            ctx->scale_x = dp.scale_x;
            ctx->scale_y = dp.scale_y;

            ctx->ui = std::move(dp.ui_frame);
            ctx->inf = std::move(dp.inf_frame);

            // branch
            infer_in_.push_drop_oldest(ctx); // global pool
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

            res.bboxes = run_inference_(ctx->inf);
            ctx->inf.release();

            auto it = pipes_by_stream_id_.find(ctx->stream_id);
            if (it != pipes_by_stream_id_.end() && it->second) {
                it->second->inf_res.push_drop_oldest(std::move(res));
            }
        }
    }

    void PipelineRuntime::anonymizer_loop_(StreamPipe* pipe) {
        if (!pipe) return;

        std::unordered_map<int64_t, FramePtr> pending_frames;
        std::unordered_map<int64_t, InferResults> pending_infer_results;

        while (running_.load(std::memory_order_relaxed)) {
            // drain results
            FramePtr ctx;
            while (pipe->anon_in.try_pop(ctx)) {
                if (!ctx) continue;
                const int64_t id = ctx->frame_id;
                pending_frames[id] = ctx;

                auto it = pending_infer_results.find(id);
                if (it != pending_infer_results.end()) {
                    auto res = std::move(it->second);
                    pending_infer_results.erase(it);

                    anonymize_(ctx->ui, res.bboxes, ctx->scale_x, ctx->scale_y);
                    pipe->enc_in.push_drop_oldest(ctx);
                    pending_frames.erase(id);
                }
            }
            // drain res
            InferResults r;
            while (pipe->inf_res.try_pop(r)) {
                const int64_t id = r.frame_id;
                auto it = pending_frames.find(id);
                if (it != pending_frames.end()) {
                    auto ctx2 = it->second;

                    anonymize_(ctx2->ui, r.bboxes, ctx2->scale_x, ctx2->scale_y);
                    pipe->enc_in.push_drop_oldest(ctx2);
                    pending_frames.erase(id);
                } else {
                    pending_infer_results[id] = std::move(r);
                }
            }

            if (pending_frames.size() > 500) pending_frames.clear();
            if (pending_infer_results.size() > 500) pending_infer_results.clear();

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
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
                "\"h\":" + std::to_string(ctx->ui.rows) +
                "}";
            server_.push_meta(ui_key, std::move(ui_meta));

            std::string inf_meta =
                "{"
                "\"stream_id\":\"" + ctx->stream_id + "\","
                "\"profile\":\"inf\","
                "\"frame_id\":" + std::to_string(ctx->frame_id) + ","
                "\"pts_ns\":" + std::to_string(ctx->pts_ns) +
                "}";
            server_.push_meta(inf_key, std::move(inf_meta));
        }
    }

    // hooks

    std::vector<Box> PipelineRuntime::run_inference_(const cv::Mat& /* inf */) {
        return {};
    }

    void PipelineRuntime::anonymize_(cv::Mat& /* ui */,
                                     const std::vector<Box>& /* bboxes */,
                                     float /* sx */, float /* sy */) {

    }
}
