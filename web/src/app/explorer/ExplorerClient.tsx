"use client";

import { useState } from "react";
import { HormoneTrajectoryChart } from "@/components/chart/HormoneTrajectoryChart";
import { LiveInference } from "@/components/LiveInference";
import type { Participant, ParticipantIndex } from "@/lib/types";
import type { Explanation } from "@/lib/data";

type Layer = "model" | "truth" | "calendar";

export function ExplorerClient({
  index,
  participants,
  explanations,
}: {
  index: ParticipantIndex[];
  participants: Record<string, Participant>;
  explanations: Record<string, Explanation>;
}) {
  const [pid, setPid] = useState(index[0]?.pid ?? "");
  const [layers, setLayers] = useState<Set<Layer>>(new Set(["truth", "calendar"]));
  const [showRefusal, setShowRefusal] = useState(false);
  const p = participants[pid];
  const meta = index.find((x) => x.pid === pid);
  const exp = explanations[pid];

  const toggle = (l: Layer) =>
    setLayers((prev) => {
      const next = new Set(prev);
      if (next.has(l)) next.delete(l);
      else next.add(l);
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
    <div className="max-w-shell mx-auto px-6 py-12">
      <div className="flex items-baseline justify-between flex-wrap gap-3 mb-8">
        <div>
          <p className="eyebrow mb-2">Trajectory explorer</p>
          <h1 className="heading text-[30px]">Reconstructed hormone curves</h1>
        </div>
        {meta && (
          <div className="flex items-center gap-2">
            <span className="px-3 py-1.5 rounded-pill border border-hair text-[12px] text-ink-muted">
              cycle-length σ <span className="num text-ink-secondary">{meta.cycleLenStd}d</span>
            </span>
            <span
              className={`px-3 py-1.5 rounded-pill text-[12px] ${
                meta.regularity === "irregular"
                  ? "bg-accent-dim text-accent border border-[rgba(217,238,80,0.4)]"
                  : "border border-hair text-ink-muted"
              }`}
            >
              {meta.regularity}
            </span>
          </div>
        )}
      </div>

      {/* controls */}
      <div className="flex flex-wrap items-center gap-2 mb-7">
        <select
          value={pid}
          onChange={(e) => setPid(e.target.value)}
          className="pill pill-ghost appearance-none cursor-pointer pr-10"
          style={{
            backgroundImage:
              "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%23888888' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E\")",
            backgroundRepeat: "no-repeat",
            backgroundSize: "11px 8px",
            backgroundPosition: "right 15px center",
          }}
        >
          {index.map((x) => (
            <option key={x.pid} value={x.pid} className="bg-raised">
              {x.pid} · {x.regularity} · {x.nCycles} cycles
            </option>
          ))}
        </select>

        <div className="w-px h-6 bg-hair mx-1" />

        {(["truth", "model", "calendar"] as Layer[]).map((l) => (
          <button
            key={l}
            onClick={() => toggle(l)}
            className={`pill ${layers.has(l) ? "pill-on" : "pill-ghost"}`}
          >
            {l}
          </button>
        ))}
      </div>

      {/* the hero chart */}
      <div className="card p-5 md:p-7">
        {p && <HormoneTrajectoryChart data={p} layers={layers} />}

        <div className="mt-6 flex flex-wrap items-end gap-8 border-t border-hair pt-5">
          <div className="text-[13px] text-ink-muted max-w-md leading-relaxed">
            <Key color="var(--truth)" /> ground truth (urine assay) ·{" "}
            <Key color="var(--s-pdg)" /> model estimate · <Key color="var(--calendar)" dashed /> calendar.
            Shaded band = 90% prediction interval. Synthetic participant (CC-BY).
            <br />
            <span className="opacity-60">
              The calendar drawn here is the naive fixed day-14 rule (b1a), because that is what a
              period app actually shows you. Every skill number on /skill is scored against the much
              stronger hierarchical hazard baseline (b1c), so these two figures are not comparable.
            </span>
          </div>
          <div className="flex gap-10 ml-auto">
            {calMae != null && (
              <Metric tone="critical" value={calMae} label="day-14 calendar MAE (b1a)" />
            )}
            {modelMae != null && layers.has("model") && (
              <Metric tone="accent" value={modelMae} label="model ovulation MAE" />
            )}
          </div>
        </div>
      </div>

      {/* grounded explanation */}
      {exp && (
        <div className="card p-7 mt-4">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <p className="eyebrow">Grounded explanation</p>
            <span className="px-3 py-1.5 rounded-pill bg-accent-dim text-accent text-[12px]">
              {exp.nCited} of {exp.nCited} claims cited · 0 diagnostic claims · every number rendered from model output, not written by the model
            </span>
          </div>
          <p className="text-[15px] leading-[1.7] text-ink-secondary">{exp.text}</p>
          <div className="flex flex-wrap gap-1.5 mt-5">
            {exp.citations.map((c) => (
              <span
                key={c.id}
                title={`${c.claim} — ${c.source}`}
                className="px-2.5 py-1 rounded-pill border border-hair text-[11px] text-ink-muted hover:border-strong hover:text-ink-secondary transition-colors cursor-help"
              >
                {c.id}
              </span>
            ))}
          </div>
          <div className="mt-6 pt-5 border-t border-hair">
            <button onClick={() => setShowRefusal((s) => !s)} className="pill pill-ghost text-[13px]">
              {showRefusal ? "▾" : "▸"} try an unsafe question: “{exp.refusalDemo.question}”
            </button>
            {showRefusal && (
              <p className="text-[14px] text-warning mt-4 leading-[1.7] border-l-2 border-warning pl-4">
                {exp.refusalDemo.answer}
              </p>
            )}
          </div>
        </div>
      )}

      {/* live backend */}
      <LiveInference />
    </div>
  );
}

function Key({ color, dashed }: { color: string; dashed?: boolean }) {
  return (
    <span
      className="inline-block w-3.5 align-middle mr-1"
      style={{
        borderTop: `2px ${dashed ? "dashed" : "solid"} ${color}`,
      }}
    />
  );
}

function Metric({ tone, value, label }: { tone: "critical" | "accent"; value: number; label: string }) {
  return (
    <div className="text-right">
      <div
        className="num text-[34px] leading-none tracking-[-0.03em] font-medium"
        style={{ color: `var(--${tone})` }}
      >
        {value.toFixed(1)}
        <span className="text-[17px] text-ink-muted ml-0.5">d</span>
      </div>
      <div className="eyebrow mt-2">{label}</div>
    </div>
  );
}
