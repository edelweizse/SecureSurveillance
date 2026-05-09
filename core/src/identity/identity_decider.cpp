#include <identity/identity_decider.hpp>

#include <stdexcept>
#include <utility>

namespace veilsight {
    namespace {
        class NoopIdentityDecider final : public IIdentityDecider {
        public:
            IdentityResult decide(const IdentityTask& task) override {
                IdentityResult out;
                out.stream_id = task.stream_id;
                out.frame_id = task.frame_id;
                out.frame = task.frame;
                out.tracks = task.tracks;
                for (auto& track : out.tracks) {
                    track.identity_key.clear();
                    track.identity_confidence = 0.0f;
                    track.privacy_action = "anonymize";
                }
                if (out.frame) {
                    out.frame->tracked_boxes = out.tracks;
                }
                return out;
            }
        };

        class NoopIdentityDeciderFactory final : public IIdentityDeciderFactory {
        public:
            explicit NoopIdentityDeciderFactory(IdentityModuleConfig cfg)
                : cfg_(std::move(cfg)) {}

            std::unique_ptr<IIdentityDecider> create() const override {
                return std::make_unique<NoopIdentityDecider>();
            }

            int backend_threads() const override {
                return 1;
            }

        private:
            IdentityModuleConfig cfg_;
        };
    }

    std::unique_ptr<IIdentityDeciderFactory> create_identity_decider_factory(const IdentityModuleConfig& cfg) {
        if (cfg.type.empty() || cfg.type == "noop" || cfg.type == "none") {
            return std::make_unique<NoopIdentityDeciderFactory>(cfg);
        }
        throw std::invalid_argument("[Identity] Unsupported identity decider type: " + cfg.type);
    }

    std::unique_ptr<IIdentityDecider> create_identity_decider(const IdentityModuleConfig& cfg) {
        return create_identity_decider_factory(cfg)->create();
    }
}
