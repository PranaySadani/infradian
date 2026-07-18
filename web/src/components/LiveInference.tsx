"use client";

import { useState } from "react";
import { api, API_BASE, type DayRecord, type TrajectoryResponse } from "@/lib/api";

/** Live end-to-end call into the FastAPI backend: synthesize a wearable series, POST it to
 *  /predict/trajectory, and render what the reference model returns. This is the frontend↔backend
 *  wiring made demonstrable. It is strictly additive — every other view reads static JSON, so if the
 *  API is down the page is unaffected. */
export function LiveInference() {
  const [state, setState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [res, setRes] = useState<TrajectoryResponse | null>(null);
  const [err, setErr] = useState<string>("");

  const run = async () => {
    setState("loading");
    setErr("");
    try {
      // A plausible 30-day wearable series: flat follicular, then a sustained post-ovulatory
      // thermal + heart-rate elevation from ~day 15 (the progesterone signature).
      const days: DayRecord[] = Array.from({ length: 30 }, (_, i) => {
        const luteal = i >= 15;
        return {
          day_in_study: i,
          skin_temp_dev_c: +(luteal ? 0.28 + 0.03 * Math.sin(i) : 0.02 * Math.sin(i)).toFixed(3),
          rhr_bpm: +(60 + (luteal ? 2.6 : 0) + 0.6 * Math.sin(i)).toFixed(2),
          hrv_rmssd_ms: +(45 - (luteal ? 4.5 : 0) + Math.cos(i)).toFixed(2),
          resp_rate: +(15 + (luteal ? 0.25 : 0)).toFixed(2),
          sleep_eff: 0.88,
          menses_reported: i < 4 ? 1 : 0,
        };
      });
      const r = await api.predictTrajectory("live-demo", days);
      setRes(r);
      setState("done");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setState("error");
    }
  };

  const peak =
    res && res.ovulation_prob.length
      ? res.days[res.ovulation_prob.indexOf(Math.max(...res.ovulation_prob))]
      : null;

  return (
    <div className="card p-7 mt-4">
      <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
        <div>
          <p className="eyebrow mb-2">Live inference · backend</p>
          <h2 className="heading text-[19px]">Send a wearable series to the model</h2>
        </div>
        <button onClick={run} disabled={state === "loading"} className="pill pill-accent disabled:opacity-50">
          {state === "loading" ? "Running…" : "Run inference"}
        </button>
      </div>

      <p className="text-[14px] text-ink-secondary leading-[1.65] max-w-[70ch]">
        Posts a synthetic 30-day series — flat follicular, then a sustained thermal and heart-rate
        elevation from day 15 — to <span className="num text-ink-muted">POST /predict/trajectory</span>.
        The backend runs it through the <i>same</i> feature transform used in training (parity-tested)
        and returns the reference model’s hormone estimates.
      </p>

      {state === "done" && res && (
        <div className="mt-6 grid sm:grid-cols-3 gap-3">
          <Out label="model" value={res.model} />
          <Out label="days scored" value={String(res.days.length)} />
          <Out label="peak ovulation prob." value={peak != null ? `day ${peak}` : "—"} accent />
          <div className="sm:col-span-3 rounded-xl bg-raised border border-hair p-4">
            <div className="eyebrow mb-2">predicted PdG (first 12 days)</div>
            <div className="num text-[13px] text-ink-secondary break-all leading-relaxed">
              {res.pdg_pred.slice(0, 12).map((v) => v.toFixed(2)).join("  ")}
            </div>
          </div>
        </div>
      )}

      {state === "error" && (
        <div className="mt-5 rounded-xl border border-hair bg-raised p-4">
          <p className="text-[13px] text-ink-secondary leading-relaxed">
            Backend not reachable at <span className="num text-ink-muted">{API_BASE}</span> — this is a
            normal state, and nothing else on the page depends on it. Start it with{" "}
            <span className="num text-ink-muted">make serve</span>.
          </p>
          <p className="text-[12px] text-ink-muted mt-2 num">{err}</p>
        </div>
      )}
    </div>
  );
}

function Out({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-xl bg-raised border border-hair p-4">
      <div className="eyebrow mb-2">{label}</div>
      <div className={`num text-[16px] ${accent ? "text-accent" : "text-ink"}`}>{value}</div>
    </div>
  );
}
