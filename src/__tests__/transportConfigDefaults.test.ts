describe("transport config defaults", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.resetModules();
    process.env = { ...originalEnv };
    delete process.env.VITE_VIDEO_PRIMARY_TRANSPORT;
    delete process.env.REACT_APP_VIDEO_PRIMARY_TRANSPORT;
    delete process.env.VITE_VIDEO_FALLBACK_TRANSPORT;
    delete process.env.REACT_APP_VIDEO_FALLBACK_TRANSPORT;
    delete process.env.VITE_JANUS_FIRST_TRACK_TIMEOUT_MS;
    delete process.env.REACT_APP_JANUS_FIRST_TRACK_TIMEOUT_MS;
    delete process.env.VITE_WSJPEG_FPS_LOCAL;
    delete process.env.REACT_APP_WSJPEG_FPS_LOCAL;
    delete process.env.VITE_WSJPEG_FPS_REMOTE;
    delete process.env.REACT_APP_WSJPEG_FPS_REMOTE;
    delete process.env.VITE_WSJPEG_ADAPTIVE;
    delete process.env.REACT_APP_WSJPEG_ADAPTIVE;
    delete process.env.VITE_WSJPEG_RENDER_MAX_FPS;
    delete process.env.REACT_APP_WSJPEG_RENDER_MAX_FPS;
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  test("uses janus primary with wsjpeg fallback by default", () => {
    const { API } = require("../config") as typeof import("../config");
    expect(API.VIDEO_PRIMARY_TRANSPORT).toBe("janus");
    expect(API.VIDEO_FALLBACK_TRANSPORT).toBe("wsjpeg");
    expect(API.VIDEO_TRANSPORT_PLAN).toEqual(["janus", "wsjpeg"]);
  });

  test("forces janus primary and a single explicit fallback", () => {
    process.env.VITE_VIDEO_PRIMARY_TRANSPORT = "direct";
    process.env.VITE_VIDEO_FALLBACK_TRANSPORT = "auto";
    const { API } = require("../config") as typeof import("../config");
    expect(API.VIDEO_PRIMARY_TRANSPORT).toBe("janus");
    expect(API.VIDEO_FALLBACK_TRANSPORT).toBe("wsjpeg");
    expect(API.VIDEO_TRANSPORT_PLAN).toEqual(["janus", "wsjpeg"]);
  });

  test("supports explicit fallback disable without adding extra hops", () => {
    process.env.VITE_VIDEO_FALLBACK_TRANSPORT = "none";
    const { API } = require("../config") as typeof import("../config");
    expect(API.VIDEO_TRANSPORT_PLAN).toEqual(["janus"]);
  });

  test("applies transport timeout and fallback stream tuning from env", () => {
    process.env.VITE_JANUS_FIRST_TRACK_TIMEOUT_MS = "19000";
    process.env.VITE_WSJPEG_FPS_LOCAL = "25";
    process.env.VITE_WSJPEG_FPS_REMOTE = "9";
    process.env.VITE_WSJPEG_ADAPTIVE = "0";
    process.env.VITE_WSJPEG_RENDER_MAX_FPS = "14";
    const { API } = require("../config") as typeof import("../config");
    expect(API.JANUS_FIRST_TRACK_TIMEOUT_MS).toBe(19000);
    expect(API.WSJPEG_FPS_LOCAL).toBe(25);
    expect(API.WSJPEG_FPS_REMOTE).toBe(9);
    expect(API.WSJPEG_ADAPTIVE).toBe(false);
    expect(API.WSJPEG_RENDER_MAX_FPS).toBe(14);
    expect(API.VIDEO_TRANSPORT_PLAN).toEqual(["janus", "wsjpeg"]);
  });
});
