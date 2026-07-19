export const dynamic = "error";

export default function ImpactPage() {
  return (
    <div className="max-w-[900px] mx-auto px-6 py-20">
      <p className="eyebrow mb-6">Impact · who this is for and what it changes</p>
      <h1 className="display text-[clamp(32px,4.6vw,52px)] max-w-[19ch]">
        Measurement is the bottleneck, and it is the cheapest thing to fix.
      </h1>

      <p className="mt-8 text-[17px] leading-[1.65] text-ink-secondary max-w-[68ch]">
        I want to be precise about what this project does and does not do for people, because the
        honest version is more useful than the inflated one.
      </p>

      {/* Sarah */}
      <section className="card p-8 mt-12">
        <p className="eyebrow mb-4">The person this is built around</p>
        <h2 className="heading text-[22px] mb-4">Sarah, 34, cycles between 24 and 41 days</h2>
        <div className="text-[15px] leading-[1.7] text-ink-secondary space-y-4">
          <p>
            Three apps tell her she ovulates on day 14. All three are using the same arithmetic and
            none of them have looked at her body. She has been trying to conceive for eleven months.
            At her appointment the doctor asks when she actually ovulates, and she has no answer,
            because a blood test is one point and her hormones are a curve.
          </p>
          <p>
            What she can produce cheaply is a wearable she already owns, a sentence about how she
            feels, and an at-home test strip. What she cannot produce is a defensible answer. That
            gap is not a modelling problem. It is a measurement problem, and it is what this
            benchmark exists to close.
          </p>
        </div>
        <div className="mt-6 grid sm:grid-cols-4 gap-2">
          {[
            ["Speak or type", "a sentence about the day"],
            ["Photograph", "an at-home hormone test"],
            ["See", "the reconstructed trajectory"],
            ["Bring", "one page to the appointment"],
          ].map(([a, b]) => (
            <div key={a} className="rounded-xl bg-raised border border-hair p-4">
              <div className="text-[14px] text-accent">{a}</div>
              <div className="text-[12px] text-ink-muted mt-1 leading-snug">{b}</div>
            </div>
          ))}
        </div>
      </section>

      {/* reach */}
      <section className="mt-4 grid md:grid-cols-3 gap-4">
        <Stat
          value="~1 in 10"
          label="Women of reproductive age affected by PCOS, the most common endocrine disorder in this group."
          src="WHO"
        />
        <Stat
          value="Years, not weeks"
          label="Typical delay between first symptoms and a diagnosis for endometriosis, reported consistently across health systems."
          src="widely reported clinical estimate"
        />
        <Stat
          value="~$1T"
          label="Annual global economic opportunity from closing the women's health gap, the figure quoted in this challenge brief."
          src="Hack-Nation Challenge 05 brief"
        />
      </section>

      <p className="text-[13px] text-ink-muted mt-4 leading-relaxed max-w-[76ch]">
        I have deliberately not invented a precision figure for how many people this helps. The two
        clinical numbers above are widely reported rather than derived by me, and the third is quoted
        from the brief itself. Anything more specific would be a number I could not defend.
      </p>

      {/* what changes */}
      <section className="card p-8 mt-12">
        <h2 className="heading text-[22px] mb-5">What this actually changes today</h2>
        <ul className="space-y-4 text-[15px] leading-[1.65] text-ink-secondary">
          <Item n="1" title="A shared ruler where there was none">
            Cycle-inference claims are currently unfalsifiable, because there is no agreed task,
            metric or split. A frozen spec with a named metric and participant-disjoint splits makes
            two labs comparable for the first time. That is the contribution that compounds.
          </Item>
          <Item n="2" title="A hard floor under future claims">
            Any product claiming to predict ovulation from a wearable can now be asked one question:
            what is your skill over a properly specified calendar baseline, stratified by regularity.
            My own answer on real data is a null, which is exactly the point. The bar is now public.
          </Item>
          <Item n="3" title="Symptom logs that can be pooled">
            INFRADIAN-SYM is an open vocabulary mapping free text onto the same schema columns the
            model consumes. Symptom data is the cheapest longitudinal signal in this field and the
            least reusable. Fixing the vocabulary is unglamorous and it is what makes pooling possible.
          </Item>
          <Item n="4" title="Zero access friction">
            The clinical data needs a data use agreement, which excludes most people who would
            otherwise contribute. The CC-BY synthetic cohort means anyone can run the entire benchmark
            today, and the reference checkpoint is trained only on it precisely so it can be shared.
          </Item>
        </ul>
      </section>

      {/* honest limits */}
      <section className="card p-8 mt-4" style={{ borderColor: "rgba(217,238,80,0.35)" }}>
        <h2 className="heading text-[22px] mb-5 text-accent">What it does not do, stated plainly</h2>
        <ul className="space-y-3 text-[15px] leading-[1.65] text-ink-secondary list-none">
          <li>
            It does not help anyone clinically today. On real consumer wearables the pre-registered
            primary endpoint is a null. Nobody should change a decision because of this model.
          </li>
          <li>
            The clinical cohort is 42 Canadian adults aged 18 to 29. It says nothing about
            perimenopause, nothing about PCOS-typical populations, and nothing about older or more
            diverse groups, which are the populations with the most to gain.
          </li>
          <li>
            It is not a diagnostic device and it is emphatically not contraception. Wearable-derived
            cycle estimates are not reliable enough to prevent or plan a pregnancy, and the
            explanation layer refuses questions that head that way.
          </li>
        </ul>
        <p className="text-[14px] text-ink-secondary mt-6 leading-[1.7]">
          The reach argument here is not that a model helps a million women tomorrow. It is that the
          field cannot get there without a ruler, the ruler did not exist, and now it does, along
          with the first honest reading off it.
        </p>
      </section>
    </div>
  );
}

function Stat({ value, label, src }: { value: string; label: string; src: string }) {
  return (
    <div className="card p-6">
      <div className="text-[26px] font-medium tracking-[-0.02em] text-accent leading-tight">{value}</div>
      <div className="text-[13px] text-ink-secondary mt-3 leading-[1.55]">{label}</div>
      <div className="text-[11px] text-ink-muted mt-3">{src}</div>
    </div>
  );
}

function Item({ n, title, children }: { n: string; title: string; children: React.ReactNode }) {
  return (
    <li className="flex gap-4">
      <span className="num text-[13px] text-accent shrink-0 mt-1">{n}</span>
      <span>
        <span className="text-ink block mb-1">{title}</span>
        {children}
      </span>
    </li>
  );
}
