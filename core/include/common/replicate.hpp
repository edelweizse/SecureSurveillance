#pragma once

#include <common/config.hpp>

#include <vector>

namespace veilsight {
    std::vector<IngestConfig> expand_replicas(const std::vector<IngestConfig>& in);
}
