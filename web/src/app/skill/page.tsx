import { skill } from "@/lib/data";

export const dynamic = "error";

export default function SkillPage() {
  const s = skill();
  const maxAbs = 0.55;

  return (
    <div className="max-w-[1000px] mx-auto px-6 py-20">
      <p className="eyebrow mb-6">Skill over Calendar · stratified by cycle regularity</p>
      <h1 className="display text-[clamp(34px,5vw,56px)] max-w-[18ch]">
        The gain is concentrated where the calendar <span className="text-accent">fails</span>.
      </h1>
      <p className="mt-7 text-[17px] leading-[1.6] text-ink-secondary max-w-[64ch]">
        We built the strongest calendar baseline we could — an empirical-Bayes hazard model, not a
        fixed day-14 strawman — and made it the denominator on purpose. On the synthetic cohort the
        model barely beats it for regular cycles and beats it decisively for irregular ones. On real
        consumer wearables the primary endpoint is a <span className="text-ink">null</span>, and we say
        so here rather than burying it.
      </p>

      <div className="card p-7 md:p-9 mt-12 space-y-9">
        {s.strata.map((row) => (
          <div key={row.stratum} className="grid sm:grid-cols-[150px_1fr] gap-x-6 gap-y-3 items-center">
            <div>
              <div className="text-[16px] capitalize font-medium tracking-[-0.01em]">{row.stratum}</div>
              {row.synthetic.primary && (
                <div className="mt-1 inline-block px-2.5 py-1 rounded-pill bg-accent-dim text-accent text-[11px] font-medium">
                  pre-registered primary
                </div>
              )}
            </div>
            <div className="space-y-2.5">
              <SkillBar label="synthetic" cell={row.synthetic} maxAbs={maxAbs} />
              <SkillBar label="real (mcPHASES)" cell={row.real} maxAbs={maxAbs} muted />
            </div>
          </div>
        ))}
      </div>

      <p className="text-[13px] text-ink-muted mt-6 leading-relaxed max-w-[80ch]">
        SoC = 1 − model_error / calendar_error; positive means the model beats the calendar. Whiskers
        are 95% cluster-bootstrap confidence intervals over participants; p from a Wilcoxon signed-rank
        test on per-participant differences. The mean of per-participant skill ratios is never reported
        — it is the unstable quantity.
      </p>

      <section className="mt-16 card p-8">
        <h2 className="heading text-[22px] mb-4">Why a null here is the right result</h2>
        <p className="text-[15px] leading-[1.65] text-ink-secondary max-w-[72ch]">
          The temperature rise that a wearable detects is a <em>consequence</em> of ovulation —
          progesterone from the corpus luteum drives it 1–3 days <em>after</em> the event. So a wearable
          cannot forecast ovulation ahead of it, and a benchmark that claimed otherwise would be the
          suspicious one. What we can measure is whether it beats the calendar at confirming timing, and
          on real consumer-grade data at n=42 it does not. Cycle-phase information is clearly present
          (macro-F1 0.46 against 0.25 chance) — the day-level signal-to-noise simply is not there yet.
        </p>
      </section>
    </div>
  );
}

function SkillBar({
  label,
  cell,
  maxAbs,
  muted,
}: {
  label: string;
  cell: { soc: number | null; lo: number | null; hi: number | null; n: number; p: number | null };
  maxAbs: number;
  muted?: boolean;
}) {
  const soc = cell.soc ?? 0;
  const pct = (v: number) => `${((v / maxAbs) * 50 + 50).toFixed(1)}%`;
  const positive = soc >= 0;
  const color = muted ? "var(--ink-muted)" : positive ? "var(--accent)" : "var(--critical)";

  return (
    <div className="flex items-center gap-3">
      <div className="w-[104px] text-[12px] text-ink-muted shrink-0">{label}</div>
      <div className="relative flex-1 h-8 rounded-lg bg-raised border border-hair overflow-hidden">
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-strong" />
        {cell.lo != null && cell.hi != null && (
          <div
            className="absolute top-1/2 h-px bg-ink-muted"
            style={{
              left: pct(Math.min(cell.lo, cell.hi)),
              right: `calc(100% - ${pct(Math.max(cell.lo, cell.hi))})`,
            }}
          />
        )}
        <div
          className="absolute top-1/2 -translate-y-1/2 h-3.5 rounded-sm"
          style={{
            background: color,
            left: positive ? "50%" : pct(soc),
            width: `calc(${pct(soc)} - 50%)`,
          }}
        />
      </div>
      <div className="w-[112px] text-right num text-[13px] shrink-0" style={{ color }}>
        {soc >= 0 ? "+" : ""}
        {soc.toFixed(2)}
        {cell.p != null && <span className="text-ink-muted text-[11px]"> p={cell.p}</span>}
      </div>
    </div>
  );
}
