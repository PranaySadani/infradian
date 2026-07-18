import { skill, leaderboard } from "@/lib/data";

export const dynamic = "error";

export default function SkillPage() {
  const s = skill();
  const board = leaderboard();
  const maxAbs = 0.55;

  return (
    <div className="max-w-[1000px] mx-auto px-6 py-12">
      <p className="eyebrow mb-3">Skill over Calendar · reported by cycle regularity</p>
      <h1 className="text-[32px] font-semibold tracking-[-0.02em] mb-2">
        The gain is concentrated where the calendar fails.
      </h1>
      <p className="text-[15px] text-ink-secondary max-w-2xl leading-relaxed mb-10">
        We built the strongest calendar baseline we could — an empirical-Bayes hazard model, not a fixed
        day-14 strawman — and made it the denominator on purpose. On the synthetic cohort, the model barely
        beats it for regular cycles and beats it decisively for irregular ones. On real consumer wearables,
        the primary endpoint is a <span className="text-ink">null</span> — and we say so.
      </p>

      <div className="space-y-8">
        {s.strata.map((row) => (
          <div key={row.stratum} className="grid sm:grid-cols-[120px_1fr] gap-4 items-center">
            <div>
              <div className="text-[14px] font-medium capitalize">{row.stratum}</div>
              {row.synthetic.primary && <div className="eyebrow text-warning">pre-registered primary</div>}
            </div>
            <div className="space-y-2">
              <SkillBar label="synthetic" cell={row.synthetic} maxAbs={maxAbs} />
              <SkillBar label="real (mcPHASES)" cell={row.real} maxAbs={maxAbs} muted />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 mono text-[11px] text-ink-muted">
        SoC = 1 − model_error / calendar_error. Positive = model beats calendar. Bars show 95% cluster
        bootstrap CI over participants; p from Wilcoxon signed-rank on per-participant differences.
      </div>

      <section className="mt-16">
        <p className="eyebrow mb-4">Full benchmark · synthetic vs real vs sim-to-real transfer</p>
        <div className="overflow-x-auto rounded-lg border border-hair">
          <table className="w-full text-[13px]">
            <thead className="text-ink-muted eyebrow">
              <tr className="border-b border-hair">
                <th className="text-left px-4 py-3">Task</th>
                <th className="text-right px-4 py-3">Synthetic</th>
                <th className="text-right px-4 py-3">Real</th>
                <th className="text-right px-4 py-3">Sim→Real</th>
              </tr>
            </thead>
            <tbody className="mono">
              {board.map((r) => (
                <tr key={r.task} className="border-b border-hair last:border-0">
                  <td className="px-4 py-3 font-sans text-ink-secondary">
                    {r.label} <span className="text-ink-muted">· {r.metric}</span>
                  </td>
                  <td className="px-4 py-3 text-right">{fmt(r.synthetic)}</td>
                  <td className="px-4 py-3 text-right text-ink">{fmt(r.real)}</td>
                  <td className="px-4 py-3 text-right text-ink-muted">{fmt(r.transfer)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-[12px] text-ink-muted mt-3 max-w-2xl leading-snug">
          Idealized physiology (synthetic) does not survive contact with real consumer wearables: phase
          information is present (macro-F1 0.46 &gt; 0.25 chance) but day-level hormone reconstruction and
          ovulation timing are not solved. The dilution negative control is near-zero on real data, so the
          little signal that exists is not a hydration artifact.
        </p>
      </section>
    </div>
  );
}

function fmt(v: number | null) {
  return v == null ? "—" : (v >= 0 ? "+" : "") + v.toFixed(3);
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
  const color = muted ? "var(--ink-muted)" : positive ? "var(--good)" : "var(--critical)";
  return (
    <div className="flex items-center gap-3">
      <div className="w-28 text-[11px] text-ink-muted mono shrink-0">{label}</div>
      <div className="relative flex-1 h-7 bg-surface rounded border border-hair">
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-strong" />
        {/* CI whisker */}
        {cell.lo != null && cell.hi != null && (
          <div
            className="absolute top-1/2 h-px bg-ink-muted"
            style={{ left: pct(Math.min(cell.lo, cell.hi)), right: `calc(100% - ${pct(Math.max(cell.lo, cell.hi))})` }}
          />
        )}
        <div
          className="absolute top-1/2 -translate-y-1/2 h-3 rounded-sm"
          style={{
            background: color,
            left: positive ? "50%" : pct(soc),
            width: `calc(${pct(soc)} - 50%)`,
            transform: positive ? "translateY(-50%)" : "translateY(-50%)",
          }}
        />
      </div>
      <div className="w-24 text-right mono text-[12px] shrink-0" style={{ color }}>
        {soc >= 0 ? "+" : ""}
        {soc.toFixed(2)}
        {cell.p != null && <span className="text-ink-muted text-[10px]"> p={cell.p}</span>}
      </div>
    </div>
  );
}
