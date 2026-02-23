import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { IdentityDetail } from "../pages/IdentityDetail";
import { Library } from "../pages/Library";

describe("Identity card navigation", () => {
  test("clicking a card opens detail route", async () => {
    const user = userEvent.setup();
    const sample = [
      {
        id: "42",
        first_seen: "2026-02-22T12:00:00Z",
        last_seen: "2026-02-22T14:00:00Z",
        face_samples: ["/faces/42_face.jpg"],
        body_samples: [],
        stats: { frequency: 9 }
      }
    ];

    render(
      <MemoryRouter initialEntries={["/library"]}>
        <Routes>
          <Route path="/library" element={<Library initialIdentities={sample} />} />
          <Route path="/library/:id" element={<IdentityDetail />} />
        </Routes>
      </MemoryRouter>
    );

    await user.click(screen.getByRole("button", { name: /open identity 42/i }));
    expect(await screen.findByText(/Identity 42/i)).toBeInTheDocument();
  });
});
