#include <recognizer/recognizer.hpp>

#include <stdexcept>
#include <utility>

namespace veilsight {
    std::unique_ptr<IRecognizerFactory> create_mobilefacenet_recognizer_factory(
        const RecognizerModuleConfig& cfg);

    namespace {
        class NoopRecognizer final : public IRecognizer {
        public:
            RecognitionResult recognize(const RecognitionTask& task) override {
                RecognitionResult out;
                out.stream_id = task.stream_id;
                out.frame_id = task.frame_id;
                out.frame = task.frame;
                out.tracks = task.tracks;
                return out;
            }
        };

        class NoopRecognizerFactory final : public IRecognizerFactory {
        public:
            explicit NoopRecognizerFactory(RecognizerModuleConfig cfg)
                : cfg_(std::move(cfg)) {}

            std::unique_ptr<IRecognizer> create() const override {
                return std::make_unique<NoopRecognizer>();
            }

            int backend_threads() const override {
                return 1;
            }

        private:
            RecognizerModuleConfig cfg_;
        };
    }

    std::unique_ptr<IRecognizerFactory> create_recognizer_factory(const RecognizerModuleConfig& cfg) {
        if (cfg.type.empty() || cfg.type == "noop" || cfg.type == "none") {
            return std::make_unique<NoopRecognizerFactory>(cfg);
        }
        if (cfg.type == "mobilefacenet") {
            return create_mobilefacenet_recognizer_factory(cfg);
        }
        throw std::invalid_argument("[Recognizer] Unsupported recognizer type: " + cfg.type);
    }

    std::unique_ptr<IRecognizer> create_recognizer(const RecognizerModuleConfig& cfg) {
        return create_recognizer_factory(cfg)->create();
    }
}
