export const dynamic = "error";

export default function MethodsPage() {
  return (
    <div className="max-w-[820px] mx-auto px-6 py-12 leading-relaxed">
      <p className="eyebrow mb-3">Methods · limitations · what we can and cannot claim</p>
      <h1 className="text-[32px] font-semibold tracking-[-0.02em] mb-8">
        This page is a credibility asset, not fine print.
      </h1>

      <Section title="The three-tier, license-correct design">
        <p>
          <b>Tier A (open):</b> NHANES-derived tables — US public domain, released CC0.{" "}
          <b>Tier B (gated):</b> mcPHASES — PhysioNet Restricted Health Data License 1.5.0. We publish{" "}
          <i>no</i> raw rows, no per-participant figures, and no mcPHASES-trained checkpoint — only a loader,
          SHA-256 checksums, hashed per-fold participant IDs, and aggregate metrics. <b>Tier C (synthetic):</b>{" "}
          a CC-BY cohort anyone can run the whole benchmark on with zero data-access friction. Every
          trajectory plotted in this app is a Tier C synthetic participant.
        </p>
      </Section>

      <Section title="What we measured, honestly">
        <ul className="list-disc pl-5 space-y-1.5 text-ink-secondary">
          <li>
            One <b>pre-registered</b> primary endpoint (committed before any real-data run): T2-R ovulation
            timing, irregular stratum, paired difference vs the hazard calendar. Everything else is labelled
            exploratory — no cherry-picking a lucky cell out of ~100.
          </li>
          <li>
            The calendar baseline is deliberately <b>strong</b> (empirical-Bayes hazard model), so a win means
            something.
          </li>
          <li>
            Every feature is causal, enforced by a future-perturbation test. No participant spans a fold. The
            20 dual-round participants are grouped by identity so they never leak across folds.
          </li>
          <li>
            A urine-dilution <b>negative control</b> (PdG on follicular days only) bounds how much apparent
            signal could be a hydration artifact. On real data it is ~0.02 — negligible.
          </li>
        </ul>
      </Section>

      <Section title="The result, stated plainly">
        <p>
          On the synthetic cohort the model beats the calendar exactly where the calendar fails (irregular
          cycles, SoC +0.40, p&lt;0.001). On <b>real</b> consumer wearables the primary endpoint is a{" "}
          <b>null</b> (SoC −0.07, p=0.48): the day-level signal-to-noise is too low to beat the calendar for
          ovulation timing, even though cycle-phase information is clearly present (macro-F1 0.46 vs 0.25
          chance). The sim-to-real gap — idealized physiology collapsing on real data — is the finding, and
          the benchmark is the reusable instrument that makes it measurable.
        </p>
      </Section>

      <Section title="What this is NOT">
        <ul className="list-disc pl-5 space-y-1.5 text-ink-secondary">
          <li>Not a diagnostic device. Not contraceptive or fertility guidance.</li>
          <li>
            The real cohort is 42 Canadian adults, ages 18–29, no hormonal contraception, no diabetes. It says{" "}
            <b>nothing</b> about perimenopause, PCOS-typical populations, or older or more diverse groups.
          </li>
          <li>
            Phase labels in mcPHASES come from a proprietary algorithm; we build headline claims on measured
            hormone values, not those labels.
          </li>
          <li>An unfixable seasonal-temperature confound exists (absolute dates were stripped); we ablate for it and report it.</li>
        </ul>
      </Section>

      <Section title="Artifacts">
        <ul className="list-disc pl-5 space-y-1.5 text-ink-secondary mono text-[13px]">
          <li>Benchmark spec + harness (Apache-2.0)</li>
          <li>INFRADIAN-SYNTH-1K synthetic cohort (CC-BY-4.0)</li>
          <li>infradian-ref-s reference checkpoint, synthetic-trained (Apache-2.0)</li>
          <li>Frozen split manifest + checksums for gated Tier B reproduction</li>
        </ul>
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-10">
      <h2 className="text-[18px] font-semibold mb-3">{title}</h2>
      <div className="text-[15px] text-ink-secondary">{children}</div>
    </section>
  );
}
