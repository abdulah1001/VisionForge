import { NavLink, Outlet } from "react-router-dom";
import { Activity, Boxes, History } from "lucide-react";
import { VisionForgeLogo } from "./VisionForgeLogo";
import { ThemeSwitcher } from "./ThemeSwitcher";
import { useJobs } from "@/api/hooks";

const links = [
  ["/studio", "Studio", Boxes],
  ["/jobs", "Jobs", History],
  ["/system", "System", Activity],
] as const;

export function AppShell() {
  const jobs = useJobs();
  const active =
    jobs.data?.filter((j) =>
      ["queued", "running", "cancelling"].includes(j.status),
    ).length ?? 0;
  const gpuBusy = active > 0;

  return (
    <div className="surface-noise min-h-screen overflow-x-hidden">
      <header className="sticky top-0 z-40 border-b border-[var(--b1)] bg-[color-mix(in_srgb,var(--bg)_84%,transparent)] backdrop-blur-2xl">
        <div className="mx-auto flex h-[68px] max-w-[1600px] items-center gap-2 px-3 sm:gap-5 sm:px-4 lg:px-8">
          <VisionForgeLogo />
          <div
            className="ml-1 hidden items-center gap-2 rounded-full border border-[var(--b1)] bg-[var(--s1)] px-2.5 py-1 text-[10px] text-[var(--dim)] shadow-[var(--shadow-soft)] md:flex"
            title={gpuBusy ? "GPU job active" : "GPU idle"}
          >
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{
                background: gpuBusy ? "var(--warning)" : "var(--success)",
                boxShadow: `0 0 0 3px color-mix(in srgb, ${gpuBusy ? "var(--warning)" : "var(--success)"} 25%, transparent)`,
              }}
              aria-hidden
            />
            <span className="mono uppercase tracking-wider">
              {gpuBusy ? `GPU busy · ${active}` : "GPU idle"}
            </span>
          </div>
          <nav className="ml-auto flex min-w-0 items-center gap-0.5 sm:gap-1" aria-label="App">
            {links.map(([to, label, Icon]) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `nav-chip ${isActive ? "is-active" : "text-[var(--muted)] hover:text-[var(--text)]"}`
                }
              >
                <Icon size={16} aria-hidden />
                <span className="hidden sm:inline">{label}</span>
              </NavLink>
            ))}
          </nav>
          <ThemeSwitcher />
        </div>
      </header>
      <main className="min-w-0">
        <Outlet />
      </main>
    </div>
  );
}
