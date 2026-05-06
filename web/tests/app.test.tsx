import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { WebRTCPlayer } from "../src/components/WebRTCPlayer";

const sockets: MockSocket[] = [];

class MockSocket {
  onmessage: ((event: MessageEvent) => void) | null = null;
  close = vi.fn();
  constructor(public url: string) {
    sockets.push(this);
  }
  emit(value: unknown) {
    this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent);
  }
}

global.WebSocket = MockSocket as any;

class MockMediaStream {
  constructor(public tracks: unknown[] = []) {}
}

class MockPeerConnection {
  static instances: MockPeerConnection[] = [];
  iceGatheringState = "complete";
  connectionState = "new";
  localDescription = { type: "offer", sdp: "mock-offer" };
  ontrack: ((event: RTCTrackEvent) => void) | null = null;
  onconnectionstatechange: (() => void) | null = null;
  addTransceiver = vi.fn();
  addEventListener = vi.fn();
  close = vi.fn(() => {
    this.connectionState = "closed";
  });
  createOffer = vi.fn(async () => ({ type: "offer", sdp: "mock-offer" }));
  setLocalDescription = vi.fn(async () => undefined);
  setRemoteDescription = vi.fn(async () => undefined);

  constructor() {
    MockPeerConnection.instances.push(this);
  }
}

global.MediaStream = MockMediaStream as any;

function stream(stream_id: string) {
  return {
    stream_id,
    profile: "ui",
    webrtc_available: false,
    webrtc_offer_url: "",
    mjpeg_url: `http://localhost:8080/video/${stream_id}/ui`,
    snapshot_url: "",
    metadata_url: ""
  };
}

function rule(name = "Entry line") {
  return {
    id: "rule-1",
    stream_id: "cam0",
    profile: "ui",
    kind: "line",
    name,
    enabled: true,
    geometry: { points: [{ x: 10, y: 10 }, { x: 50, y: 50 }] },
    settings: {},
    created_at_ms: 1,
    updated_at_ms: 1
  };
}

function snapshot(stream_id: string, frame_id = 5, rules: any[] = []) {
  return {
    stream: { stream_id, profile: "ui" },
    frame_id,
    timestamp_ms: 1000,
    width: 100,
    height: 100,
    counts: { active_tracks: 1, unique_tracks: 1 },
    tracks: [{ id: 1, bbox: { x: 10, y: 10, w: 20, h: 30 }, foot: { x: 20, y: 38 }, dwell_s: 1.2, velocity_px_s: 3 }],
    heatmap: { rows: 2, cols: 2, values: [1, 0, 0, 0], max_value: 1 },
    density: { rows: 2, cols: 2, values: [1, 0, 0, 0], max_value: 1 },
    directions: [],
    routes: [],
    rules,
    recent_events: []
  };
}

function mockFetch(streams = [stream("cam0")]) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    if (url === "/api/streams") return Response.json({ public_base_url: "http://localhost:8080", streams });
    if (url === "/api/config") return Response.json({ path: "/tmp/config.yaml", yaml_text: "streams: []" });
    if (url === "/api/pipeline/status") return Response.json({ state: "stopped" });
    if (url === "/api/health") return Response.json({ ok: true, connection_state: "connected", runner: { ok: true } });
    if (url === "/api/metrics") return Response.json({});
    if (url === "/api/pipeline/start") return Response.json({ accepted: true, status: { state: "running" } });
    if (url === "/api/analytics/rules" && init?.method === "POST") {
      const body = JSON.parse(String(init.body));
      return Response.json({ id: "rule-1", created_at_ms: 1, updated_at_ms: 1, enabled: true, settings: {}, ...body });
    }
    if (String(url).startsWith("/api/analytics/rules/") && init?.method === "PUT") {
      const body = JSON.parse(String(init.body));
      return Response.json({ ...rule("Entry line"), updated_at_ms: 2, ...body });
    }
    if (String(url).startsWith("/api/analytics/rules/") && init?.method === "DELETE") {
      return Response.json({ deleted: true });
    }
    return Response.json({ valid: true });
  });
}

describe("dashboard", () => {
  beforeEach(() => {
    sockets.length = 0;
    MockPeerConnection.instances.length = 0;
    global.RTCPeerConnection = MockPeerConnection as any;
    HTMLMediaElement.prototype.play = vi.fn(async () => undefined);
    HTMLCanvasElement.prototype.getBoundingClientRect = () =>
      ({ left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
  });

  afterEach(() => {
    cleanup();
  });

  it("renders streams and calls pipeline commands", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;

    render(<App />);
    expect(await screen.findByText("cam0/ui")).toBeInTheDocument();
    expect(screen.getAllByText("Protocol: MJPEG").length).toBeGreaterThan(0);
    await userEvent.click(screen.getByTitle("Start"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/pipeline/start", { method: "POST" }));
  });

  it("focuses a clicked stream thumbnail", async () => {
    global.fetch = mockFetch([stream("cam0"), stream("cam1")]) as any;
    const { container } = render(<App />);

    expect(await screen.findByText("cam1/ui")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /cam1\/ui/ }));
    expect(container.querySelector(".focused-stream header strong")?.textContent).toBe("cam1/ui");
  });

  it("attaches streamless WebRTC tracks to the video element", async () => {
    const fetchMock = vi.fn(async () =>
      new Response("mock-answer", {
        status: 201,
        headers: { "Content-Type": "application/sdp", Location: "/webrtc/whep/sessions/sess-1" }
      })
    );
    global.fetch = fetchMock as any;
    const rtcStream = {
      ...stream("cam0"),
      webrtc_available: true,
      webrtc_offer_url: "http://localhost:8080/webrtc/whep/cam0/ui"
    };

    const { container } = render(<WebRTCPlayer stream={rtcStream} />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(rtcStream.webrtc_offer_url, expect.any(Object)));
    const peer = MockPeerConnection.instances[0];
    const track = { kind: "video" };
    peer.ontrack?.({ streams: [], track } as unknown as RTCTrackEvent);

    const video = container.querySelector("video") as HTMLVideoElement;
    expect((video.srcObject as unknown as MockMediaStream).tracks).toEqual([track]);
  });

  it("updates the analytics panel from overlay websocket snapshots and toggles layers", async () => {
    global.fetch = mockFetch() as any;
    render(<App />);

    expect(await screen.findByText("cam0/ui")).toBeInTheDocument();
    sockets.find((socket) => socket.url.endsWith("/ws/analytics/overlays"))?.emit(snapshot("cam0", 12));
    await waitFor(() => expect(screen.getByText("12")).toBeInTheDocument());

    const heatmap = screen.getByLabelText("Heatmap") as HTMLInputElement;
    expect(heatmap.checked).toBe(false);
    await userEvent.click(heatmap);
    expect(heatmap.checked).toBe(true);
  });

  it("submits line rule geometry in frame coordinates", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;
    render(<App />);

    expect(await screen.findByText("cam0/ui")).toBeInTheDocument();
    sockets.find((socket) => socket.url.endsWith("/ws/analytics/overlays"))?.emit(snapshot("cam0"));
    await userEvent.click(screen.getByTitle("Line rule"));
    const canvas = document.querySelector("canvas.analytics-overlay");
    expect(canvas).not.toBeNull();
    fireEvent.click(canvas as HTMLCanvasElement, { clientX: 10, clientY: 10 });
    fireEvent.click(canvas as HTMLCanvasElement, { clientX: 50, clientY: 50 });

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/analytics/rules", expect.any(Object)));
    const createCall = fetchMock.mock.calls.find(([url]) => url === "/api/analytics/rules");
    const body = JSON.parse(String(createCall?.[1]?.body));
    expect(body.geometry.points).toEqual([{ x: 10, y: 10 }, { x: 50, y: 50 }]);
  });

  it("renames and deletes drawn analytical events", async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock as any;
    render(<App />);

    expect(await screen.findByText("cam0/ui")).toBeInTheDocument();
    sockets.find((socket) => socket.url.endsWith("/ws/analytics/overlays"))?.emit(snapshot("cam0", 5, [rule("Entry line")]));
    expect(await screen.findByText("Entry line")).toBeInTheDocument();

    await userEvent.click(screen.getByLabelText("Rename Entry line"));
    await userEvent.clear(screen.getByLabelText("Rename Entry line"));
    await userEvent.type(screen.getByLabelText("Rename Entry line"), "Lobby line");
    await userEvent.click(screen.getByLabelText("Save Entry line"));

    await waitFor(() => expect(screen.getByText("Lobby line")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith("/api/analytics/rules/rule-1", expect.objectContaining({ method: "PUT" }));

    await userEvent.click(screen.getByLabelText("Delete Lobby line"));
    await waitFor(() => expect(screen.queryByText("Lobby line")).not.toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith("/api/analytics/rules/rule-1", { method: "DELETE" });
  });
});
