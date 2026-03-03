describe("transport config defaults", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.resetModules();
    process.env = { ...originalEnv };
    delete process.env.VITE_VIDEO_PRIMARY_TRANSPORT;
    delete process.env.REACT_APP_VIDEO_PRIMARY_TRANSPORT;
    delete process.env.VITE_VIDEO_FALLBACK_TRANSPORT;
    delete process.env.REACT_APP_VIDEO_FALLBACK_TRANSPORT;
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
});
