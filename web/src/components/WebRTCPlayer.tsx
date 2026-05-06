import { useEffect, useRef, useState } from "react";
import type { StreamInfo } from "../api/client";

export type StreamProtocol = "WebRTC" | "MJPEG";

async function waitForIce(peer: RTCPeerConnection, timeoutMs: number): Promise<void> {
  if (peer.iceGatheringState === "complete") return;
  await new Promise<void>((resolve) => {
    const timer = window.setTimeout(resolve, timeoutMs);
    peer.addEventListener("icegatheringstatechange", () => {
      if (peer.iceGatheringState === "complete") {
        window.clearTimeout(timer);
        resolve();
      }
    });
  });
}

export function WebRTCPlayer({
  stream,
  preferWebRTC = true,
  onProtocolChange
}: {
  stream: StreamInfo;
  preferWebRTC?: boolean;
  onProtocolChange?: (protocol: StreamProtocol) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const markVideoReadyRef = useRef<(() => void) | null>(null);
  const [failed, setFailed] = useState(!stream.webrtc_available || !preferWebRTC);
  const protocol: StreamProtocol = failed ? "MJPEG" : "WebRTC";

  useEffect(() => {
    const canUseWebRTC = preferWebRTC && stream.webrtc_available && typeof RTCPeerConnection !== "undefined";
    setFailed(!canUseWebRTC);
    if (!canUseWebRTC) return;

    let closed = false;
    let sessionUrl: string | null = null;
    let sawVideo = false;
    const videoTimeout = window.setTimeout(() => {
      if (!sawVideo) {
        deleteSession();
        setFailed(true);
        peer.close();
      }
    }, 5000);
    const peer = new RTCPeerConnection();
    function deleteSession() {
      if (!sessionUrl) return;
      const url = sessionUrl;
      sessionUrl = null;
      void fetch(url, { method: "DELETE", keepalive: true }).catch(() => {});
    }

    peer.addTransceiver("video", { direction: "recvonly" });
    peer.ontrack = (event) => {
      if (videoRef.current) {
        videoRef.current.srcObject = event.streams[0] ?? new MediaStream([event.track]);
        void videoRef.current.play().catch(() => {});
      }
    };
    peer.onconnectionstatechange = () => {
      if (["failed", "closed", "disconnected"].includes(peer.connectionState)) {
        deleteSession();
        setFailed(true);
      }
    };

    function markVideoReady() {
      sawVideo = true;
      window.clearTimeout(videoTimeout);
    }
    markVideoReadyRef.current = markVideoReady;

    function closePeer() {
      closed = true;
      window.clearTimeout(videoTimeout);
      peer.close();
      deleteSession();
    }

    window.addEventListener("pagehide", closePeer);

    async function connect() {
      try {
        const offer = await peer.createOffer();
        await peer.setLocalDescription(offer);
        await waitForIce(peer, 1500);
        const response = await fetch(stream.webrtc_offer_url, {
          method: "POST",
          headers: { "Content-Type": "application/sdp" },
          body: peer.localDescription?.sdp ?? offer.sdp ?? ""
        });
        if (!response.ok) throw new Error(`WHEP failed: ${response.status}`);
        const location = response.headers.get("Location");
        if (location) sessionUrl = new URL(location, stream.webrtc_offer_url).toString();
        const answer = await response.text();
        if (closed) {
          deleteSession();
          return;
        }
        await peer.setRemoteDescription({ type: "answer", sdp: answer });
      } catch {
        deleteSession();
        setFailed(true);
        peer.close();
      }
    }

    void connect();
    return () => {
      window.removeEventListener("pagehide", closePeer);
      markVideoReadyRef.current = null;
      closePeer();
    };
  }, [preferWebRTC, stream]);

  useEffect(() => {
    onProtocolChange?.(protocol);
  }, [onProtocolChange, protocol]);

  return (
    <div className="player-frame">
      {failed ? (
        <img className="stream-media" src={stream.mjpeg_url} alt={`${stream.stream_id} MJPEG`} />
      ) : (
        <video
          ref={videoRef}
          className="stream-media"
          autoPlay
          playsInline
          muted
          onLoadedData={() => markVideoReadyRef.current?.()}
          onPlaying={() => markVideoReadyRef.current?.()}
        />
      )}
      <span className="protocol-badge">Protocol: {protocol}</span>
    </div>
  );
}
