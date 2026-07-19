// Backend wiring. The demo path is static JSON, so the API is strictly additive:
// it powers live inference and synthetic sampling. Every call degrades gracefully.

// Default to same-origin `/api`, which is where the Vercel Python function lives in production.
// Override with NEXT_PUBLIC_API_URL to point at a separately hosted backend (e.g. the local
// FastAPI server on :8000 during development).
export const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "/api").replace(/\/$/, "");

export interface DayRecord {
  day_in_study: number;
  skin_temp_dev_c?: number | null;
  rhr_bpm?: number | null;
  hrv_rmssd_ms?: number | null;
  resp_rate?: number | null;
  sleep_eff?: number | null;
  menses_reported?: number;
}

export interface TrajectoryResponse {
  participant_id: string;
  days: number[];
  pdg_pred: number[];
  e3g_pred: number[];
  ovulation_prob: number[];
  model: string;
}

async function call<T>(path: string, init?: RequestInit, timeoutMs = 8000): Promise<T> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: ctrl.signal,
      headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    });
    if (!r.ok) throw new Error(`${path} -> ${r.status}`);
    return (await r.json()) as T;
  } finally {
    clearTimeout(t);
  }
}

export const api = {
  health: () => call<{ status: string; model_loaded: boolean }>("/healthz"),
  model: () => call<Record<string, unknown>>("/model"),
  predictTrajectory: (participant_id: string, days: DayRecord[]) =>
    call<TrajectoryResponse>("/predict/trajectory", {
      method: "POST",
      body: JSON.stringify({ participant_id, days }),
    }),
  syntheticSample: (n = 1, days = 90, seed = 0) =>
    call<{ n: number; rows: number; sample: Record<string, unknown>[] }>(
      `/synthetic/sample?n=${n}&days=${days}&seed=${seed}`
    ),
};
