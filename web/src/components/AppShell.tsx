import Link from "next/link";
import { ApiStatus } from "./ApiStatus";

const NAV = [
  { href: "/", label: "Thesis" },
  { href: "/explorer", label: "Explorer" },
  { href: "/journal", label: "Journal" },
  { href: "/skill", label: "Result" },
  { href: "/impact", label: "Impact" },
  { href: "/methods", label: "Methods" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-20 px-4 pt-4">
        <div className="max-w-shell mx-auto flex items-center justify-between gap-4 rounded-pill border border-hair bg-surface/80 backdrop-blur-xl px-5 py-2.5">
          <Link href="/" className="flex items-center gap-2.5 shrink-0">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-accent" />
            <span className="text-[15px] font-semibold tracking-[-0.02em]">INFRADIAN</span>
          </Link>

          <nav className="hidden md:flex items-center gap-1">
            {NAV.map((n) => (
              <Link
                key={n.href}
                href={n.href}
                className="px-3.5 py-1.5 rounded-pill text-[14px] text-ink-secondary hover:text-ink hover:bg-white/5 transition-colors"
              >
                {n.label}
              </Link>
            ))}
          </nav>

          <div className="flex items-center gap-2.5 shrink-0">
            <ApiStatus />
            <a
              href="https://github.com/PranaySadani/infradian"
              target="_blank"
              rel="noreferrer"
              className="pill pill-accent hidden sm:inline-block"
            >
              Get the benchmark
            </a>
          </div>
        </div>
      </header>

      <main className="flex-1">{children}</main>

      <footer className="mt-24 px-4 pb-6">
        <div className="max-w-shell mx-auto rounded-card border border-hair bg-surface px-6 py-5">
          <div className="flex flex-wrap gap-x-8 gap-y-2 text-[13px] text-ink-muted">
            <span className="text-ink-secondary">Not a diagnostic device.</span>
            <span>Clinical cohort n=42, Canadian, ages 18–29 — says nothing about menopause or PCOS populations.</span>
            <span className="num">Apache-2.0 · CC-BY-4.0 · CC0</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
