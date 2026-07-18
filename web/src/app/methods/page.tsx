export const dynamic = "error";

export default function MethodsPage() {
  return (
    <div className="max-w-[880px] mx-auto px-6 py-20">
      <p className="eyebrow mb-6">Methods · limitations · what we can and cannot claim</p>
      <h1 className="display text-[clamp(32px,4.6vw,50px)] max-w-[20ch]">
        This page is a credibility asset, not fine print.
      </h1>

      <div className="mt-14 space-y-4">
        <Card title="The three-tier, license-correct design">
          <p>
            <b className="text-ink">Tier A (open)</b> — NHANES-derived tables, US public domain,
            released CC0. <b className="text-ink">Tier B (gated)</b> — mcPHASES, under the PhysioNet
            Restricted Health Data License 1.5.0: we publish <i>no</i> raw rows, no per-participant
            figures, and no mcPHASES-trained checkpoint — only a loader, SHA-256 checksums, hashed
            per-fold participant IDs, and aggregate metrics. <b className="text-ink">Tier C
            (synthetic)</b> — a CC-BY cohort anyone can run the whole benchmark on with zero access
            friction. Every trajectory plotted in this app is a Tier C synthetic participant.
          </p>
        </Card>

        <Card title="What we measured, honestly">
          <ul className="space-y-2.5 list-none">
            <Bullet>
              One <b className="text-ink">pre-registered</b> primary endpoint, committed to git before
              any real-data run: ovulation timing, irregular stratum, paired difference against the
              hazard calendar. Everything else is labelled exploratory — no cherry-picking a lucky cell
              out of roughly a hundred.
            </Bullet>
            <Bullet>
              The calendar baseline is deliberately <b className="text-ink">strong</b> (empirical-Bayes
              hazard model), so a win would mean something — and so a loss is informative.
            </Bullet>
            <Bullet>
              Every feature is causal, enforced by a future-perturbation and truncation-invariance test.
              No participant spans a fold; the 20 dual-round participants are grouped by identity.
            </Bullet>
            <Bullet>
              A urine-dilution <b className="text-ink">negative control</b> (PdG on follicular days only)
              bounds how much apparent signal could be a hydration artifact. On real data it is ≈0.02.
            </Bullet>
          </ul>
        </Card>

        <Card title="The result, stated plainly" accent>
          <p>
            On the synthetic cohort the model beats the calendar exactly where the calendar fails —
            irregular cycles, SoC +0.40, p&lt;0.001. On <b className="text-ink">real</b> consumer
            wearables the primary endpoint is a <b className="text-ink">null</b> (SoC −0.07, p=0.48):
            the day-level signal-to-noise is too low to beat the calendar for ovulation timing, even
            though cycle-phase information is clearly present (macro-F1 0.46 against 0.25 chance). The
            sim-to-real gap is the finding, and the benchmark is the reusable instrument that makes it
            measurable.
          </p>
        </Card>

        <Card title="What this is NOT">
          <ul className="space-y-2.5 list-none">
            <Bullet>Not a diagnostic device. Not contraceptive or fertility guidance.</Bullet>
            <Bullet>
              The real cohort is 42 Canadian adults, ages 18–29, no hormonal contraception, no diabetes.
              It says <b className="text-ink">nothing</b> about perimenopause, PCOS-typical populations,
              or older and more diverse groups.
            </Bullet>
            <Bullet>
              Phase labels in mcPHASES come from a proprietary algorithm; headline claims are built on
              measured hormone values and LH-derived ovulation, not those labels.
            </Bullet>
            <Bullet>
              An unfixable seasonal-temperature confound exists — mcPHASES stripped absolute dates — so
              we ablate for it and report it rather than pretending it away.
            </Bullet>
          </ul>
        </Card>

        <Card title="Artifacts">
          <div className="flex flex-wrap gap-2">
            {[
              "Benchmark spec + harness · Apache-2.0",
              "INFRADIAN-SYNTH-1K · CC-BY-4.0",
              "infradian-ref-s checkpoint · Apache-2.0",
              "Split manifest + checksums",
              "Grounded explanation layer + eval",
            ].map((t) => (
              <span key={t} className="px-3.5 py-1.5 rounded-pill border border-hair text-[13px] text-ink-secondary">
                {t}
              </span>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function Card({
  title,
  children,
  accent,
}: {
  title: string;
  children: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <section
      className="card p-8"
      style={accent ? { borderColor: "rgba(217,238,80,0.35)" } : undefined}
    >
      <h2 className={`heading text-[20px] mb-4 ${accent ? "text-accent" : ""}`}>{title}</h2>
      <div className="text-[15px] leading-[1.68] text-ink-secondary">{children}</div>
    </section>
  );
}

function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="text-accent mt-[7px] shrink-0 w-1 h-1 rounded-full bg-accent" />
      <span>{children}</span>
    </li>
  );
}
