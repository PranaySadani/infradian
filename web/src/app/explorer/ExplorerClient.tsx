"use client";

import { useState } from "react";
import { HormoneTrajectoryChart } from "@/components/chart/HormoneTrajectoryChart";
import type { Participant, ParticipantIndex } from "@/lib/types";

type Layer = "model" | "truth" | "calendar";

export function ExplorerClient({
  index,
  participants,
}: {
  index: ParticipantIndex[];
  participants: Record<string, Participant>;
}) {
  const [pid, setPid] = useState(index[0]?.pid ?? "");
  const [layers, setLayers] = useState<Set<Layer>>(new Set(["truth", "calendar"]));
  const p = participants[pid];
  const meta = index.find((x) => x.pid === pid);

  const toggle = (l: Layer) =>
    setLayers((prev) => {
      const next = new Set(prev);
      next.has(l) ? next.delete(l) : next.add(l);
      return next;
    });

  const ovMae = (kind: "calendarDay" | "modelDay") => {
    if (!p?.events.length) return null;
    const errs = p.events.map((e) => Math.abs(e[kind] - e.truthDay));
    return errs.reduce((a, b) => a + b, 0) / errs.length;
  };
  const calMae = ovMae("calendarDay");
  const modelMae = ovMae("modelDay");

  return (
    <div className="max-w-[1280px] mx-auto px-6 py-8">
      <div className="flex flex-wrap items-center gap-4 mb-6">
        <div className="flex items-center gap-2">
          <label className="eyebrow">participant</label>
          <select
            value={pid}
            onChange={(e) => setPid(e.target.value)}
            className="bg-surface border border-strong rounded-md px-3 py-1.5 text-[13px] mono"
          >
            {index.map((x) => (
              <option key={x.pid} value={x.pid}>
                {x.pid} · {x.regularity} · {x.nCycles} cycles
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-1.5">
          {(["truth", "model", "calendar"] as Layer[]).map((l) => (
            <button
              key={l}
              onClick={() => toggle(l)}
              className={`px-3 py-1.5 rounded-md text-[12px] border transition-colors ${
                layers.has(l)
                  ? "border-strong bg-raised text-ink"
                  : "border-hair text-ink-muted hover:text-ink-secondary"
              }`}
            >
              {l}
            </button>
          ))}
        </div>
        {meta && (
          <span className="mono text-[11px] text-ink-muted ml-auto">
            cycle-length σ = {meta.cycleLenStd}d {meta.regularity === "irregular" ? "· ragged" : "· regular"}
          </span>
        )}
      </div>

      {p && <HormoneTrajectoryChart data={p} layers={layers} />}

      <div className="mt-6 flex flex-wrap items-end gap-8 border-t border-hair pt-4">
        <div className="text-[12px] text-ink-secondary max-w-md leading-snug">
          <span className="inline-block w-3 h-[2px] align-middle bg-truth mr-1" /> ground truth (urine assay) ·{" "}
          <span className="inline-block w-3 h-[2px] align-middle bg-pdg mr-1" /> model estimate ·{" "}
          <span className="inline-block w-3 h-[2px] align-middle bg-calendar mr-1" /> calendar. Shaded = 90%
          prediction interval. Synthetic participant (CC-BY).
        </div>
        <div className="flex gap-8 ml-auto">
          {calMae != null && (
            <Metric tone="critical" value={calMae} label="calendar ovulation MAE" />
          )}
          {modelMae != null && layers.has("model") && (
            <Metric tone="good" value={modelMae} label="model ovulation MAE" />
          )}
        </div>
      </div>
    </div>
  );
}

function Metric({ tone, value, label }: { tone: "critical" | "good"; value: number; label: string }) {
  return (
    <div className="text-right">
      <div className="mono text-[30px] leading-none" style={{ color: `var(--${tone})` }}>
        {value.toFixed(1)}
        <span className="text-[16px] text-ink-muted">d</span>
      </div>
      <div className="eyebrow mt-1">{label}</div>
    </div>
  );
}
