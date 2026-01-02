#pragma once

#include <memory>

#include <common/config.hpp>
#include <ingest/frame_source.hpp>

namespace ss {
    std::unique_ptr<IFrameSource> make_frame_source(const IngestConfig& cfg);
}