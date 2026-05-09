#include <face_detector/face_detector.hpp>

#include <face_detector/scrfd_detector.hpp>
#include <face_detector/yunet_detector.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace veilsight {
    namespace {
        class SCRFDFaceDetectorFactory final : public IFaceDetectorFactory {
        public:
            explicit SCRFDFaceDetectorFactory(SCRFDModuleConfig cfg)
                : cfg_(std::move(cfg)) {}

            std::unique_ptr<IFaceDetector> create() const override {
                return std::make_unique<SCRFDDetector>(cfg_);
            }

            int backend_threads() const override {
                return std::max(1, cfg_.ncnn_threads);
            }

        private:
            SCRFDModuleConfig cfg_;
        };

        class YuNetFaceDetectorFactory final : public IFaceDetectorFactory {
        public:
            explicit YuNetFaceDetectorFactory(YuNetModuleConfig cfg)
                : cfg_(std::move(cfg)) {}

            std::unique_ptr<IFaceDetector> create() const override {
                return std::make_unique<YuNetDetector>(cfg_);
            }

            int backend_threads() const override {
                return std::max(1, cfg_.ncnn_threads);
            }

        private:
            YuNetModuleConfig cfg_;
        };
    }

    std::unique_ptr<IFaceDetectorFactory> create_face_detector_factory(
        const FaceDetectorModuleConfig& cfg) {
        if (cfg.type.empty() || cfg.type == "none") {
            return nullptr;
        }
        if (cfg.type == "scrfd") {
            return std::make_unique<SCRFDFaceDetectorFactory>(cfg.scrfd);
        }
        if (cfg.type == "yunet") {
            return std::make_unique<YuNetFaceDetectorFactory>(cfg.yunet);
        }
        throw std::invalid_argument("[FaceDetector] Unsupported face detector type: " + cfg.type);
    }

    std::unique_ptr<IFaceDetector> create_face_detector(const FaceDetectorModuleConfig& cfg) {
        auto factory = create_face_detector_factory(cfg);
        return factory ? factory->create() : nullptr;
    }
}
