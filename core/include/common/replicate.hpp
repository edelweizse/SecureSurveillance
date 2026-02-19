#pragma once

#include <common/config.hpp>

#include <vector>

namespace ss {
    std::vector<IngestConfig> expand_replicas(const std::vector<IngestConfig>& in);
}
