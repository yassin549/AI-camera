describe("api client split deployment behavior", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.resetModules();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
    jest.restoreAllMocks();
  });

  test("builds metadata WS URL from remote API base and appends api_key", () => {
    process.env.VITE_API_BASE = "https://api.example.com";
    process.env.VITE_API_KEY = "topsecret";
    const client = require("../api/client") as typeof import("../api/client");

    expect(client.resolveMetadataWsUrl()).toBe(
      "wss://api.example.com/api/realtime/ws?api_key=topsecret"
    );
  });

  test("sends x-api-key header and resolves media URLs against remote API base", async () => {
    process.env.VITE_API_BASE = "https://api.example.com";
    process.env.VITE_API_KEY = "topsecret";

    const payload = [
      {
        id: 7,
        first_seen: "2026-02-24T10:00:00Z",
        last_seen: "2026-02-24T10:01:00Z",
        face_samples: ["/media/faces/7_1.jpg"],
        body_samples: ["/media/body/7_1.jpg"]
      }
    ];

    const fetchMock = jest
      .fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>()
      .mockResolvedValue({
        ok: true,
        status: 200,
        headers: {
          get: (name: string) =>
            name.toLowerCase() === "content-type" ? "application/json" : null
        },
        json: async () => payload,
        text: async () => JSON.stringify(payload)
      } as unknown as Response);
    global.fetch = fetchMock as unknown as typeof fetch;

    const client = require("../api/client") as typeof import("../api/client");
    const identities = await client.getIdentities();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("https://api.example.com/api/identities");

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    const headers = init.headers as Headers;
    expect(headers.get("x-api-key")).toBe("topsecret");

    expect(identities[0].id).toBe("7");
    expect(identities[0].face_samples).toEqual(["https://api.example.com/media/faces/7_1.jpg"]);
    expect(identities[0].body_samples).toEqual(["https://api.example.com/media/body/7_1.jpg"]);
  });
});
