import Link from "next/link";
import { skill } from "@/lib/data";

export const dynamic = "error"; // a stray dynamic call fails the BUILD, not the demo

export default function Landing() {
  const s = skill();
  const irregular = s.strata.find((x) => x.stratum === "irregular");
  return (
    <div className="max-w-[1280px] mx-auto px-6">
      <section className="pt-24 pb-16 max-w-3xl">
        <p className="eyebrow mb-6">Challenge 05 · Foundation Models for Women’s Hormonal Health</p>
        <h1 className="text-[clamp(32px,6vw,56px)] font-semibold leading-[1.05] tracking-[-0.02em]">
          A blood test is a snapshot.
          <br />
          Hormones are a trajectory.
        </h1>
        <p className="mt-6 text-[17px] text-ink-secondary leading-relaxed">
          Everyone knows the <span className="text-ink">circadian</span> rhythm. Almost nobody can name the{" "}
          <span className="text-ink">infradian</span> one — the ~28-day cycle half the planet runs on. Every
          period app predicts it with calendar arithmetic. We built the benchmark that measures whether a
          consumer wearable can do better — and exactly where it can’t.
        </p>
        <div className="mt-8 flex gap-3">
          <Link
            href="/explorer"
            className="px-4 py-2 rounded-md border border-strong text-[14px] hover:bg-raised transition-colors"
          >
            Meet a participant →
          </Link>
          <Link
            href="/skill"
            className="px-4 py-2 rounded-md text-[14px] text-ink-secondary hover:text-ink transition-colors"
          >
            See the result
          </Link>
        </div>
      </section>

      <section className="grid sm:grid-cols-3 gap-4 pb-24">
        <StatTile
          value={irregular ? `+${irregular.synthetic.soc?.toFixed(2)}` : "—"}
          ci={irregular ? `${irregular.synthetic.lo?.toFixed(2)} to ${irregular.synthetic.hi?.toFixed(2)}` : ""}
          n={irregular?.synthetic.n ?? 0}
          label="skill over calendar, irregular cycles (synthetic)"
        />
        <StatTile
          value={irregular?.real.soc != null ? irregular.real.soc.toFixed(2) : "null"}
          ci={irregular ? `p = ${irregular.real.p}` : ""}
          n={irregular?.real.n ?? 0}
          label="same endpoint on real wearables — a null result"
          tone="muted"
        />
        <StatTile value="42" ci="Canadian · ages 18–29" n={192} label="participants · complete cycles (mcPHASES)" />
      </section>

      <section className="border-t border-hair py-12 max-w-3xl">
        <p className="eyebrow mb-3">The contribution</p>
        <p className="text-[15px] text-ink-secondary leading-relaxed">
          The only checkpoint we can legally distribute is trained on <span className="text-ink">synthetic</span>{" "}
          data — mcPHASES is DUA-restricted. So we made the synthetic cohort good enough to run the whole
          benchmark on with zero data-access friction, and measured exactly how much skill it keeps on real
          clinical data. That sim-to-real gap is the finding: on clean physiology the model beats the calendar
          where the calendar fails; on real consumer wearables the day-level signal is too noisy — and now
          there is a ruler that says so, with pre-registered endpoints and participant-level confidence
          intervals.
        </p>
      </section>
    </div>
  );
}

function StatTile({
  value,
  ci,
  n,
  label,
  tone,
}: {
  value: string;
  ci: string;
  n: number;
  label: string;
  tone?: "muted";
}) {
  return (
    <div className="rounded-lg border border-hair bg-surface p-5">
      <div className={`mono text-[40px] leading-none ${tone === "muted" ? "text-ink-muted" : "text-ink"}`}>
        {value}
      </div>
      <div className="mono text-[11px] text-ink-muted mt-2">
        {ci} · n={n}
      </div>
      <div className="text-[12px] text-ink-secondary mt-3 leading-snug">{label}</div>
    </div>
  );
}
