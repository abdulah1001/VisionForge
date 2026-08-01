import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Eraser,
  Lock,
  Menu,
  MonitorSmartphone,
  Shield,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { VisionForgeLogo } from "@/components/layout/VisionForgeLogo";
import { ThemeSwitcher } from "@/components/layout/ThemeSwitcher";

const NAV = [
  { href: "#how", label: "How it works" },
  { href: "#privacy", label: "Privacy" },
  { href: "#studio", label: "Studio" },
];

const FLOW = [
  {
    title: "Upload",
    text: "Drop an MP4 or MOV. Processing stays on this machine.",
    icon: Upload,
  },
  {
    title: "Select",
    text: "Pause on a clear frame, analyze, then pick the object to remove.",
    icon: Sparkles,
  },
  {
    title: "Remove",
    text: "Track the object, rebuild the background, export a cleaned video.",
    icon: Eraser,
  },
];

export function LandingPage() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 18);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="relative overflow-x-hidden surface-noise">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-[var(--s1)] focus:px-3 focus:py-2"
      >
        Skip to content
      </a>

      <header
        className={`fixed inset-x-0 top-0 z-40 transition-[background,border-color,backdrop-filter] duration-300 ${
          scrolled
            ? "border-b border-[var(--b1)] bg-[color-mix(in_srgb,var(--bg)_88%,transparent)] backdrop-blur-xl"
            : "border-b border-transparent bg-transparent"
        }`}
      >
        <div className="mx-auto flex h-[68px] max-w-[1180px] items-center gap-4 px-4 lg:px-8">
          <VisionForgeLogo />
          <nav className="ml-auto hidden items-center gap-1 md:flex" aria-label="Primary">
            {NAV.map((n) => (
              <a
                key={n.href}
                href={n.href}
                className="nav-chip text-[var(--muted)] hover:text-[var(--text)]"
              >
                {n.label}
              </a>
            ))}
          </nav>
          <ThemeSwitcher />
          <Link className="btn btn-primary hidden sm:inline-flex" to="/studio">
            Open Studio
            <ArrowRight size={16} aria-hidden />
          </Link>
          <button
            type="button"
            className="btn md:hidden"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            onClick={() => setMenuOpen((v) => !v)}
          >
            {menuOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
        {menuOpen && (
          <div className="border-t border-[var(--b1)] bg-[var(--s1)] px-4 py-4 md:hidden">
            {NAV.map((n) => (
              <a
                key={n.href}
                href={n.href}
                className="block py-3 text-sm"
                onClick={() => setMenuOpen(false)}
              >
                {n.label}
              </a>
            ))}
            <Link className="btn btn-primary mt-2 w-full" to="/studio">
              Open Studio
            </Link>
          </div>
        )}
      </header>

      <main id="main">
        {/* Hero — brand first, one composition */}
        <section className="relative min-h-[100svh] pt-[68px]">
          <div className="pointer-events-none absolute inset-0 grid-bg opacity-30" />
          <div className="relative mx-auto grid max-w-[1180px] items-center gap-12 px-4 py-14 lg:grid-cols-[1.05fr_0.95fr] lg:gap-16 lg:px-8 lg:py-20">
            <div className="relative z-10">
              <p className="eyebrow reveal">Local AI Video Object Remover</p>
              <p className="display reveal reveal-delay-1 mt-4 text-[clamp(2.75rem,6.5vw,5.1rem)] text-[var(--text)]">
                Vision<span className="text-[var(--tracking)]">Forge</span>
              </p>
              <h1 className="reveal reveal-delay-2 mt-5 max-w-[18ch] text-[clamp(1.35rem,2.5vw,1.85rem)] font-medium leading-snug tracking-[-0.025em] text-[var(--muted)]">
                Erase objects from video. Keep the rest.
              </h1>
              <p className="reveal reveal-delay-3 mt-4 max-w-[32rem] text-base leading-relaxed text-[var(--dim)]">
                Upload a clip, select what should disappear, and export a cleaned MP4 —
                processed entirely on your GPU.
              </p>
              <div className="reveal reveal-delay-3 mt-9 flex flex-wrap gap-3">
                <Link className="btn btn-primary min-h-12 px-6" to="/studio">
                  Remove objects from video
                  <ArrowRight size={16} aria-hidden />
                </Link>
                <a className="btn min-h-12" href="#how">
                  See how it works
                </a>
              </div>
            </div>

            <div className="reveal reveal-delay-2 relative">
              <div className="erase-demo" aria-hidden>
                <div className="erase-demo__grid" />
                <div className="erase-demo__scene">
                  <div
                    className="erase-demo__bg-shape"
                    style={{
                      width: "42%",
                      height: "38%",
                      left: "8%",
                      top: "18%",
                      background: "linear-gradient(135deg, #3a556c, #1e3344)",
                    }}
                  />
                  <div
                    className="erase-demo__bg-shape"
                    style={{
                      width: "50%",
                      height: "34%",
                      right: "4%",
                      bottom: "12%",
                      background: "linear-gradient(200deg, #2d4558, #162430)",
                    }}
                  />
                  <div className="erase-demo__object" />
                  <div className="erase-demo__scan" />
                </div>
                <div className="erase-demo__badge">
                  <span>Live</span>
                  Object removal preview
                </div>
              </div>
              <p className="sr-only">
                Animated preview of an object fading out of a video frame as a scan passes over it.
              </p>
            </div>
          </div>
        </section>

        <section id="how" className="landing-section scroll-mt-24">
          <div className="mx-auto max-w-[1180px] px-4 lg:px-8">
            <p className="eyebrow">Workflow</p>
            <h2 className="display mt-3 max-w-[18ch] text-3xl md:text-4xl">
              Three steps. One clean export.
            </h2>
            <p className="mt-4 max-w-xl text-[var(--muted)]">
              No cloud upload. No model dashboard. Just a focused studio for local object removal.
            </p>

            <div className="feature-row mt-12">
              {FLOW.map((step, i) => {
                const Icon = step.icon;
                return (
                  <motion.article
                    key={step.title}
                    initial={{ opacity: 0, y: 18 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-40px" }}
                    transition={{ duration: 0.45, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] }}
                  >
                    <div className="icon-well">
                      <Icon size={18} aria-hidden />
                    </div>
                    <p className="mono text-[10px] text-[var(--dim)]">
                      {String(i + 1).padStart(2, "0")}
                    </p>
                    <h3 className="mt-2 text-lg font-semibold tracking-[-0.02em]">{step.title}</h3>
                    <p className="mt-2 text-sm leading-relaxed text-[var(--muted)]">{step.text}</p>
                  </motion.article>
                );
              })}
            </div>
          </div>
        </section>

        <section id="privacy" className="landing-section scroll-mt-24">
          <div className="mx-auto max-w-[1180px] px-4 lg:px-8">
            <p className="eyebrow">Privacy</p>
            <h2 className="display mt-3 text-3xl md:text-4xl">Local by design</h2>
            <div className="mt-10 grid gap-6 sm:grid-cols-3">
              {[
                {
                  icon: Shield,
                  title: "On-device GPU",
                  text: "Tracking and inpainting run through your local worker — footage never leaves the machine.",
                },
                {
                  icon: Lock,
                  title: "Offline models",
                  text: "Weights stay in your cache. Inference does not call model hubs.",
                },
                {
                  icon: MonitorSmartphone,
                  title: "Bound localhost",
                  text: "Studio and API default to 127.0.0.1 for a controlled, private session.",
                },
              ].map(({ icon: Icon, title, text }) => (
                <div key={title} className="panel p-5">
                  <Icon className="text-[var(--tracking)]" size={20} aria-hidden />
                  <h3 className="mt-4 text-base font-semibold">{title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-[var(--muted)]">{text}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="studio" className="landing-section scroll-mt-24">
          <div className="mx-auto max-w-[1180px] px-4 text-center lg:px-8">
            <p className="eyebrow">Studio</p>
            <h2 className="display mx-auto mt-3 max-w-[16ch] text-3xl md:text-5xl">
              Ready when your clip is.
            </h2>
            <p className="mx-auto mt-4 max-w-lg text-[var(--muted)]">
              Open the remover workspace, drop a video, and clean it in place.
            </p>
            <Link className="btn btn-primary mt-10 inline-flex min-h-12 px-7" to="/studio">
              Launch VisionForge Studio
              <ArrowRight size={16} aria-hidden />
            </Link>
          </div>
        </section>
      </main>

      <footer className="border-t border-[var(--b1)] py-12">
        <div className="mx-auto flex max-w-[1180px] flex-col gap-8 px-4 sm:flex-row sm:items-start sm:justify-between lg:px-8">
          <div>
            <VisionForgeLogo />
            <p className="mt-3 max-w-xs text-xs leading-relaxed text-[var(--dim)]">
              Local AI video object removal. ProPainter inpainting is non-commercial (NTU S-Lab).
            </p>
          </div>
          <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-[var(--muted)]">
            <Link to="/studio">Studio</Link>
            <Link to="/system">System</Link>
            <Link to="/jobs">Jobs</Link>
            <a href="#how">How it works</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
