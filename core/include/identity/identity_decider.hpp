#pragma once

#include <common/config.hpp>
#include <pipeline/tasks.hpp>

#include <memory>

namespace veilsight {
    class IIdentityDecider {
    public:
        virtual ~IIdentityDecider() = default;
        virtual IdentityResult decide(const IdentityTask& task) = 0;
    };

    class IIdentityDeciderFactory {
    public:
        virtual ~IIdentityDeciderFactory() = default;
        virtual std::unique_ptr<IIdentityDecider> create() const = 0;
        virtual int backend_threads() const = 0;
    };

    std::unique_ptr<IIdentityDeciderFactory> create_identity_decider_factory(const IdentityModuleConfig& cfg);
    std::unique_ptr<IIdentityDecider> create_identity_decider(const IdentityModuleConfig& cfg);
}
