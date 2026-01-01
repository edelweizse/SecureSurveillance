#pragma once

#include <ingest/frame_packet.hpp>

namespace ss {
    class IFrameSource {
    public:
        virtual ~IFrameSource() = default;

        virtual bool start() = 0;
        virtual bool read(FramePacket& out, int timeout_ms = 1000) = 0;
        virtual void stop() = 0;
    };
}