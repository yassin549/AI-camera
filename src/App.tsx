import { LayoutGroup } from "framer-motion";
import { useEffect } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { PremiumSwitch, type ViewMode } from "./components/PremiumSwitch";
import { IdentityDetail } from "./pages/IdentityDetail";
import { Library } from "./pages/Library";
import { LiveView } from "./pages/LiveView";

export default function App(): JSX.Element {
  const location = useLocation();
  const navigate = useNavigate();
  const pathname = location.pathname;

  const isLibraryRoute = pathname.startsWith("/library");
  const isLiveRoute = pathname.startsWith("/live") || pathname === "/";
  const mode: ViewMode = isLibraryRoute ? "library" : "live";

  useEffect(() => {
    if (pathname === "/") {
      navigate("/live", { replace: true });
      return;
    }
    if (!isLibraryRoute && !isLiveRoute) {
      navigate("/live", { replace: true });
    }
  }, [isLibraryRoute, isLiveRoute, navigate, pathname]);

  return (
    <div className="h-screen w-screen overflow-hidden">
      <header className="z-20 flex h-16 items-center justify-between border-b border-white/10 px-4 md:px-6">
        <div className="flex items-center gap-3">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-accent font-semibold text-white shadow-soft">
            A
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-100">AI Camera</div>
            <div className="text-xs text-slate-400">Low-latency overlays, raw stream preserved</div>
          </div>
        </div>
        <PremiumSwitch
          value={mode}
          onChange={(next) => navigate(next === "live" ? "/live" : "/library")}
        />
      </header>

      <main className="relative h-[calc(100vh-4rem)]">
        <LiveView active={isLiveRoute && !isLibraryRoute} />

        {isLibraryRoute ? (
          <div className="absolute inset-0">
            <LayoutGroup>
              <Routes location={location}>
                <Route path="/library" element={<Library />} />
                <Route path="/library/:id" element={<IdentityDetail />} />
                <Route path="*" element={<Navigate to="/library" replace />} />
              </Routes>
            </LayoutGroup>
          </div>
        ) : null}
      </main>
    </div>
  );
}
