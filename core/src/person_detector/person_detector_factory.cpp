#include <person_detector/person_detector.hpp>

#include <face_detector/scrfd_detector.hpp>
#include <face_detector/yunet_detector.hpp>
#include <person_detector/yolox_detector.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace veilsight {
    namespace {
        class YuNetDetectorFactory final : public IPersonDetectorFactory {
        public:
            explicit YuNetDetectorFactory(YuNetModuleConfig cfg)
                : cfg_(std::move(cfg)) {}

            std::unique_ptr<IPersonDetector> create() const override {
                return std::make_unique<YuNetDetector>(cfg_);
            }

            int backend_threads() const override {
                return std::max(1, cfg_.ncnn_threads);
            }

        private:
            YuNetModuleConfig cfg_;
        };

        class SCRFDDetectorFactory final : public IPersonDetectorFactory {
        public:
            explicit SCRFDDetectorFactory(SCRFDModuleConfig cfg)
                : cfg_(std::move(cfg)) {}

            std::unique_ptr<IPersonDetector> create() const override {
                return std::make_unique<SCRFDDetector>(cfg_);
            }

            int backend_threads() const override {
                return std::max(1, cfg_.ncnn_threads);
            }

        private:
            SCRFDModuleConfig cfg_;
        };

        class YoloXDetectorFactory final : public IPersonDetectorFactory {
        public:
            explicit YoloXDetectorFactory(YoloXModuleConfig cfg)
                : cfg_(std::move(cfg)) {}

            std::unique_ptr<IPersonDetector> create() const override {
                return std::make_unique<YoloXDetector>(cfg_);
            }

            int backend_threads() const override {
                return std::max(1, cfg_.ncnn_threads);
            }

        private:
            YoloXModuleConfig cfg_;
        };
    }

    std::unique_ptr<IPersonDetectorFactory> create_person_detector_factory(const PersonDetectorModuleConfig& cfg) {
        if (cfg.type.empty() || cfg.type == "yolox") {
            return std::make_unique<YoloXDetectorFactory>(cfg.yolox);
        }
        if (cfg.type == "yunet") {
            return std::make_unique<YuNetDetectorFactory>(cfg.yunet);
        }
        if (cfg.type == "scrfd") {
            return std::make_unique<SCRFDDetectorFactory>(cfg.scrfd);
        }
        throw std::invalid_argument("[Detector] Unsupported detector type: " + cfg.type);
    }

    std::unique_ptr<IPersonDetector> create_person_detector(const PersonDetectorModuleConfig& cfg) {
        return create_person_detector_factory(cfg)->create();
    }
}
