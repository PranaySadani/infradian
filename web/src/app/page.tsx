import Link from "next/link";
import { skill, leaderboard } from "@/lib/data";

export const dynamic = "error";

export default function Landing() {
  const s = skill();
  const board = leaderboard();
  const irregular = s.strata.find((x) => x.stratum === "irregular");
  const regular = s.strata.find((x) => x.stratum === "regular");

  return (
    <div className="max-w-shell mx-auto px-6">
      {/* hero */}
      <section className="pt-24 pb-20">
        <p className="eyebrow mb-7">Hack-Nation Challenge 05 · Women’s Hormonal Health</p>
        <h1 className="display text-[clamp(44px,7.5vw,86px)] max-w-[16ch]">
          A blood test is a snapshot.
          <br />
          Hormones are a{" "}
          <span className="text-accent">trajectory</span>.
        </h1>
        <p className="mt-8 text-[18px] leading-[1.6] text-ink-secondary max-w-[62ch]">
          Everyone knows the <span className="text-ink">circadian</span> rhythm. Almost nobody can name
          the <span className="text-ink">infradian</span> one — the ~28-day cycle half the planet runs
          on. Every period app predicts it with calendar arithmetic. We built the benchmark that
          measures whether a consumer wearable can do better — and exactly where it can’t.
        </p>
        <div className="mt-10 flex flex-wrap gap-3">
          <Link href="/explorer" className="pill pill-accent">
            Explore a participant →
          </Link>
          <Link href="/skill" className="pill pill-ghost">
            See the result
          </Link>
        </div>
      </section>

      {/* headline numbers */}
      <section className="grid md:grid-cols-3 gap-4 pb-20">
        <StatCard
          value={irregular ? `+${irregular.synthetic.soc?.toFixed(2)}` : "—"}
          accent
          detail={
            irregular
              ? `95% CI ${irregular.synthetic.lo?.toFixed(2)} – ${irregular.synthetic.hi?.toFixed(2)} · n=${irregular.synthetic.n}`
              : ""
          }
          label="Skill over calendar on irregular cycles — the pre-registered primary endpoint, on the open synthetic cohort."
        />
        <StatCard
          value={irregular?.real.soc != null ? irregular.real.soc.toFixed(2) : "null"}
          detail={irregular ? `p = ${irregular.real.p} · n=${irregular.real.n}` : ""}
          label="The same endpoint on real consumer wearables. A null result — and we lead with it."
        />
        <StatCard
          value="42"
          detail="Canadian · ages 18–29 · 192 cycles"
          label="Participants in the gated clinical cohort (mcPHASES). We redistribute none of it."
        />
      </section>

      {/* the contribution */}
      <section className="pb-20">
        <div className="card p-8 md:p-12">
          <p className="eyebrow mb-5">The contribution</p>
          <h2 className="heading text-[clamp(24px,3.2vw,36px)] max-w-[30ch]">
            The only checkpoint we can legally publish is trained on synthetic data. So we made the
            synthetic data good enough — and measured what it keeps.
          </h2>
          <p className="mt-6 text-[16px] leading-[1.65] text-ink-secondary max-w-[68ch]">
            mcPHASES is DUA-restricted, so a model trained on it cannot be distributed. We built a CC-BY
            synthetic cohort that runs the entire benchmark with zero data-access friction, published a
            reference checkpoint trained only on it, and then measured the sim-to-real gap on the real
            clinical data. That gap is the finding: on clean physiology the model beats the calendar
            exactly where the calendar fails; on real consumer wearables the day-level signal is too
            noisy — and now there is a ruler that says so, with pre-registered endpoints and
            participant-level confidence intervals.
          </p>
          <div className="mt-8 flex flex-wrap gap-2">
            {["Benchmark spec", "Synthetic cohort (CC-BY)", "Reference checkpoint", "Split manifest + checksums", "Grounded explanation layer"].map(
              (t) => (
                <span key={t} className="px-3.5 py-1.5 rounded-pill border border-hair text-[13px] text-ink-secondary">
                  {t}
                </span>
              )
            )}
          </div>
        </div>
      </section>

      {/* benchmark table */}
      <section className="pb-8">
        <div className="flex items-baseline justify-between mb-5 flex-wrap gap-2">
          <h2 className="heading text-[26px]">The full benchmark</h2>
          <p className="text-[13px] text-ink-muted">
            synthetic (reproducible by anyone) · real mcPHASES · sim-to-real transfer
          </p>
        </div>
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-[14px]">
              <thead>
                <tr className="border-b border-hair">
                  <th className="text-left font-medium text-ink-muted px-6 py-4">Task</th>
                  <th className="text-right font-medium text-ink-muted px-6 py-4">Synthetic</th>
                  <th className="text-right font-medium text-ink-muted px-6 py-4">Real</th>
                  <th className="text-right font-medium text-ink-muted px-6 py-4">Sim→Real</th>
                </tr>
              </thead>
              <tbody>
                {board.map((r) => (
                  <tr key={r.task} className="border-b border-hair last:border-0">
                    <td className="px-6 py-4">
                      <span className="text-ink-secondary">{r.label}</span>
                      <span className="text-ink-muted"> · {r.metric}</span>
                    </td>
                    <td className="px-6 py-4 text-right num text-ink-secondary">{fmt(r.synthetic)}</td>
                    <td className="px-6 py-4 text-right num text-ink">{fmt(r.real)}</td>
                    <td className="px-6 py-4 text-right num text-ink-muted">{fmt(r.transfer)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <p className="text-[13px] text-ink-muted mt-4 max-w-[76ch] leading-relaxed">
          Regular cycles show {regular ? `only +${regular.synthetic.soc?.toFixed(2)}` : "little"} skill even on
          synthetic data — the calendar is already near-optimal there. The entire gain concentrates in
          the irregular stratum, which is the equity argument, and the honest limit is that it does not
          survive contact with real consumer wearables.
        </p>
      </section>
    </div>
  );
}

function fmt(v: number | null) {
  return v == null ? "—" : (v >= 0 ? "+" : "") + v.toFixed(3);
}

function StatCard({
  value,
  detail,
  label,
  accent,
}: {
  value: string;
  detail: string;
  label: string;
  accent?: boolean;
}) {
  return (
    <div className="card p-7 flex flex-col">
      <div
        className={`num text-[clamp(38px,4.6vw,52px)] leading-none tracking-[-0.03em] font-medium ${
          accent ? "text-accent" : "text-ink"
        }`}
      >
        {value}
      </div>
      <div className="num text-[12px] text-ink-muted mt-3">{detail}</div>
      <div className="text-[14px] text-ink-secondary mt-5 leading-[1.55]">{label}</div>
    </div>
  );
}
