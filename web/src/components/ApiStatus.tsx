"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

type State = "checking" | "live" | "offline";

/** Live backend indicator. The demo never depends on the API (all views read static JSON),
 *  so "offline" is a normal, non-breaking state — it just means interactive inference is off. */
export function ApiStatus() {
  const [state, setState] = useState<State>("checking");
  const [model, setModel] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 2500);
    fetch(`${API_BASE}/healthz`, { signal: ctrl.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("bad status"))))
      .then((j) => {
        if (cancelled) return;
        setState("live");
        setModel(j.model_loaded ? "infradian-ref-s" : null);
      })
      .catch(() => !cancelled && setState("offline"))
      .finally(() => clearTimeout(t));
    return () => {
      cancelled = true;
      ctrl.abort();
      clearTimeout(t);
    };
  }, []);

  const dot =
    state === "live" ? "bg-accent" : state === "offline" ? "bg-ink-muted" : "bg-ink-muted animate-pulse";
  const label = state === "live" ? "API live" : state === "offline" ? "API offline" : "…";

  return (
    <span
      className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-pill border border-hair text-[12px] text-ink-muted"
      title={
        state === "live"
          ? `Backend reachable at ${API_BASE}${model ? ` · ${model} loaded` : ""}`
          : `Backend not reachable at ${API_BASE}. All views still work — they read static JSON.`
      }
    >
      <span className={`inline-block w-1.5 h-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  );
}
