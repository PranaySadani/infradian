import Link from "next/link";

const NAV = [
  { href: "/", label: "Thesis" },
  { href: "/explorer", label: "Explorer" },
  { href: "/skill", label: "Skill" },
  { href: "/methods", label: "Methods" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-hair sticky top-0 z-10 bg-base/90 backdrop-blur">
        <div className="max-w-[1280px] mx-auto px-6 h-14 flex items-center justify-between">
          <Link href="/" className="flex items-baseline gap-3">
            <span className="text-[15px] font-semibold tracking-tight">INFRADIAN</span>
            <span className="eyebrow hidden sm:inline">the rhythm nobody benchmarks</span>
          </Link>
          <nav className="flex items-center gap-6 text-[13px]">
            {NAV.map((n) => (
              <Link key={n.href} href={n.href} className="text-ink-secondary hover:text-ink transition-colors">
                {n.label}
              </Link>
            ))}
            <span className="mono text-[11px] flex items-center gap-1.5 text-ink-muted" title="All plotted trajectories are synthetic (CC-BY). Real mcPHASES appears only as aggregate metrics.">
              <span className="inline-block w-2 h-2 rounded-full bg-pdg" /> SYNTHETIC
            </span>
          </nav>
        </div>
      </header>
      <main className="flex-1">{children}</main>
      <footer className="border-t border-hair mt-16">
        <div className="max-w-[1280px] mx-auto px-6 py-6 text-[12px] text-ink-muted flex flex-wrap gap-x-6 gap-y-1">
          <span>Not a diagnostic device.</span>
          <span>Real cohort n=42, Canadian, ages 18–29 — says nothing about menopause or PCOS populations.</span>
          <span className="mono">Apache-2.0 · CC-BY-4.0 · CC0</span>
        </div>
      </footer>
    </div>
  );
}
