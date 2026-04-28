#pragma once
#include <memory>
#include <common/config.hpp>
#include <ingest/gst_dual_source.hpp>

namespace veilsight {
    std::unique_ptr<GstDualSource> make_dual_source(const IngestConfig& cfg);
}