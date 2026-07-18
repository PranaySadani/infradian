"use client";

import { useMemo, useState } from "react";
import { scaleLinear } from "d3-scale";
import { line, area, curveMonotoneX } from "d3-shape";
import type { Participant, HormoneKey } from "@/lib/types";

const PANELS: { key: HormoneKey; label: string; unit: string; full: string; color: string }[] = [
  { key: "e3g", label: "E3G", unit: "ng/mL", full: "Estrone-3-glucuronide", color: "var(--s-e3g)" },
  { key: "pdg", label: "PdG", unit: "µg/mL", full: "Pregnanediol glucuronide", color: "var(--s-pdg)" },
  { key: "lh", label: "LH", unit: "mIU/mL", full: "Luteinizing hormone", color: "var(--s-lh)" },
];

const M = { top: 10, right: 60, bottom: 26, left: 46 };
const PANEL_H = 104;
const GAP = 12;
const RAIL_H = 40;
const CYCLE_H = 26;
const W = 1000;

interface Props {
  data: Participant;
  layers: Set<"model" | "truth" | "calendar">;
}

export function HormoneTrajectoryChart({ data, layers }: Props) {
  const [hoverDay, setHoverDay] = useState<number | null>(null);
  const innerW = W - M.left - M.right;
  const H = M.top + PANELS.length * (PANEL_H + GAP) + RAIL_H + CYCLE_H + M.bottom;
  const days = data.days;

  const x = useMemo(
    () => scaleLinear().domain([days[0], days[days.length - 1]]).range([0, innerW]),
    [days, innerW]
  );

  return (
    <div className="relative w-full" style={{ aspectRatio: `${W} / ${H}` }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        height="100%"
        role="img"
        aria-label={`Reconstructed hormone trajectories for synthetic participant ${data.pid}`}
      >
        {PANELS.map((p, i) => {
          const top = M.top + i * (PANEL_H + GAP);
          const band = data.model[p.key];
          const truth = data.truth[p.key];
          const vals = [
            ...(band?.hi.filter((v): v is number => v != null) ?? []),
            ...truth.filter((v): v is number => v != null),
          ];
          const hi = vals.length ? Math.max(...vals) : 1;
          const y = scaleLinear().domain([0, hi * 1.12]).range([PANEL_H, 0]).nice();

          const areaGen = (lo: (number | null)[], up: (number | null)[]) =>
            area<number>()
              .defined((_, j) => lo[j] != null && up[j] != null)
              .x((_, j) => x(days[j]))
              .y0((_, j) => y(lo[j] as number))
              .y1((_, j) => y(up[j] as number))
              .curve(curveMonotoneX)(days) ?? "";

          const lineGen = (v: (number | null)[]) =>
            line<number>()
              .defined((_, j) => v[j] != null)
              .x((_, j) => x(days[j]))
              .y((_, j) => y(v[j] as number))
              .curve(curveMonotoneX)(days) ?? "";

          return (
            <g key={p.key} transform={`translate(${M.left},${top})`}>
              {y.ticks(3).map((t) => (
                <line key={t} x1={0} x2={innerW} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth={1} />
              ))}
              {layers.has("model") && band && p.key !== "lh" && (
                <path d={areaGen(band.lo, band.hi)} fill={p.color} fillOpacity={0.14} />
              )}
              {layers.has("model") && band && (
                <path
                  d={lineGen(band.mean)}
                  fill="none"
                  stroke={p.color}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity={p.key === "lh" ? 0.55 : 1}
                />
              )}
              {layers.has("truth") &&
                truth.map((v, j) =>
                  v == null ? null : (
                    <circle
                      key={j}
                      cx={x(days[j])}
                      cy={y(v)}
                      r={3}
                      fill="var(--truth)"
                      stroke="var(--bg-surface)"
                      strokeWidth={1.5}
                    />
                  )
                )}
              <text x={0} y={-1} className="eyebrow" fill="var(--ink-muted)">
                {p.label} <tspan fill="var(--ink-muted)">· {p.unit}</tspan>
              </text>
              <rect
                x={0}
                y={0}
                width={innerW}
                height={PANEL_H}
                fill="transparent"
                onPointerMove={(e) => {
                  const rect = (e.target as SVGRectElement).getBoundingClientRect();
                  const px = ((e.clientX - rect.left) / rect.width) * innerW;
                  setHoverDay(Math.round(x.invert(px)));
                }}
                onPointerLeave={() => setHoverDay(null)}
              />
            </g>
          );
        })}

        {hoverDay != null && (
          <line
            x1={M.left + x(hoverDay)}
            x2={M.left + x(hoverDay)}
            y1={M.top}
            y2={H - M.bottom}
            stroke="var(--ink-muted)"
            strokeWidth={1}
            strokeOpacity={0.6}
          />
        )}

        <OvulationEventRail
          events={data.events}
          x={x}
          y={M.top + PANELS.length * (PANEL_H + GAP)}
          offsetX={M.left}
          layers={layers}
        />
        <CycleRail
          cycles={data.cycles}
          x={x}
          y={M.top + PANELS.length * (PANEL_H + GAP) + RAIL_H}
          offsetX={M.left}
        />
        <text x={M.left} y={H - 6} className="mono" fontSize={10} fill="var(--ink-muted)">
          day in study →
        </text>
      </svg>
    </div>
  );
}

function OvulationEventRail({
  events,
  x,
  y,
  offsetX,
  layers,
}: {
  events: Participant["events"];
  x: ReturnType<typeof scaleLinear<number, number>>;
  y: number;
  offsetX: number;
  layers: Set<string>;
}) {
  return (
    <g transform={`translate(${offsetX},${y})`}>
      <text x={0} y={8} className="eyebrow" fill="var(--ink-muted)">
        ovulation
      </text>
      {events.map((e, i) => {
        const xt = x(e.truthDay);
        const xc = x(e.calendarDay);
        const xm = x(e.modelDay);
        return (
          <g key={i}>
            {layers.has("calendar") && (
              <>
                <line x1={xc} x2={xt} y1={20} y2={20} stroke="var(--critical)" strokeOpacity={0.4} strokeWidth={1.5} />
                <path d={`M${xc},15 L${xc - 4},22 L${xc + 4},22 Z`} fill="none" stroke="var(--calendar)" strokeWidth={1.2} />
              </>
            )}
            {/* truth: solid ink triangle */}
            <path d={`M${xt},14 L${xt - 4.5},23 L${xt + 4.5},23 Z`} fill="var(--truth)" />
            {layers.has("model") && (
              <path d={`M${xm},14 L${xm - 4.5},23 L${xm + 4.5},23 Z`} fill="var(--s-lh)" />
            )}
          </g>
        );
      })}
      <g transform={`translate(${x.range()[1] - 210},2)`} className="mono" fontSize={10}>
        <path d="M0,10 L-4,17 L4,17 Z" fill="var(--truth)" transform="translate(4,0)" />
        <text x={12} y={16} fill="var(--ink-muted)">truth</text>
        <path d="M60,10 L56,17 L64,17 Z" fill="none" stroke="var(--calendar)" />
        <text x={72} y={16} fill="var(--ink-muted)">calendar</text>
        <path d="M135,10 L131,17 L139,17 Z" fill="var(--s-lh)" />
        <text x={146} y={16} fill="var(--ink-muted)">model</text>
      </g>
    </g>
  );
}

function CycleRail({
  cycles,
  x,
  y,
  offsetX,
}: {
  cycles: Participant["cycles"];
  x: ReturnType<typeof scaleLinear<number, number>>;
  y: number;
  offsetX: number;
}) {
  return (
    <g transform={`translate(${offsetX},${y})`}>
      <text x={0} y={8} className="eyebrow" fill="var(--ink-muted)">
        cycles
      </text>
      {cycles.map((c) => {
        const x0 = x(c.startDay);
        const w = x(c.endDay) - x0 - 2;
        return (
          <g key={c.index}>
            <rect
              x={x0}
              y={12}
              width={Math.max(0, w)}
              height={14}
              rx={2}
              fill={c.anovulatory ? "var(--border-hair)" : "var(--border-strong)"}
              stroke={c.anovulatory ? "var(--warning)" : "none"}
              strokeOpacity={0.5}
            />
            {w > 44 && (
              <text x={x0 + w / 2} y={22} textAnchor="middle" className="mono" fontSize={10} fill="var(--ink-secondary)">
                {c.lengthDays}d
              </text>
            )}
          </g>
        );
      })}
    </g>
  );
}
