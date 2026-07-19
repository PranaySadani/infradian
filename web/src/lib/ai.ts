// Client for the multimodal AI surfaces. Every call has a defined behaviour when no OpenAI key is
// configured, so the UI never shows a dead end.

import { API_BASE } from "./api";

export interface Symptom {
  code: string;
  label: string;
  category: string;
  severity: number;
  schema_field: string | null;
  evidence: string;
}

export interface JournalResult {
  refused: boolean;
  category?: string;
  message?: string;
  text?: string;
  source?: "deterministic" | "openai";
  vocab_version?: string;
  notes?: string[];
  symptoms: Symptom[];
  schema_fields: Record<string, number>;
}

export interface StripReading {
  readable: boolean;
  analyte: string | null;
  value: number | null;
  unit: string | null;
  confidence: number;
  reason: string;
  source: string;
  warnings: string[];
  disclaimer: string;
}

export interface AiStatus {
  enabled: boolean;
  chat_model: string;
  vision_model: string;
  transcribe_model: string;
  note: string;
  features: {
    grounded_explanation: boolean;
    symptom_extraction: boolean;
    voice_transcription: boolean;
    test_strip_reading: boolean;
  };
}

async function post<T>(path: string, body: unknown, timeoutMs = 30000): Promise<T> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!r.ok) throw new Error(`${path} -> ${r.status}`);
    return (await r.json()) as T;
  } finally {
    clearTimeout(t);
  }
}

export const ai = {
  status: async (): Promise<AiStatus | null> => {
    try {
      const r = await fetch(`${API_BASE}/ai/status`);
      return r.ok ? ((await r.json()) as AiStatus) : null;
    } catch {
      return null;
    }
  },
  journal: (text: string) => post<JournalResult>("/llm/journal", { text }),
  transcribe: (audio_base64: string, filename = "clip.webm") =>
    post<{ available: boolean; text: string; message?: string }>(
      "/llm/transcribe",
      { audio_base64, filename },
      45000
    ),
  readStrip: (image_base64: string) =>
    post<StripReading>("/llm/read-strip", { image_base64 }, 45000),
};

/** Downscale an image in the browser before upload. Vercel caps request bodies near 4.5MB and a
 *  modern phone photo blows straight through that. */
export function downscaleImage(file: File, maxDim = 1000, quality = 0.75): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("could not read file"));
    reader.onload = () => {
      const img = new Image();
      img.onerror = () => reject(new Error("could not decode image"));
      img.onload = () => {
        const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
        const w = Math.round(img.width * scale);
        const h = Math.round(img.height * scale);
        const c = document.createElement("canvas");
        c.width = w;
        c.height = h;
        const ctx = c.getContext("2d");
        if (!ctx) return reject(new Error("no canvas context"));
        ctx.drawImage(img, 0, 0, w, h);
        resolve(c.toDataURL("image/jpeg", quality));
      };
      img.src = reader.result as string;
    };
    reader.readAsDataURL(file);
  });
}

export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onerror = () => reject(new Error("could not read blob"));
    r.onload = () => resolve(String(r.result).split(",")[1] ?? "");
    r.readAsDataURL(blob);
  });
}

/** Browser speech recognition, used when no OpenAI key is configured. Chrome and Safari only,
 *  which is exactly why it is the fallback rather than the primary path. */
export function browserSpeechAvailable(): boolean {
  if (typeof window === "undefined") return false;
  const w = window as unknown as Record<string, unknown>;
  return Boolean(w.SpeechRecognition || w.webkitSpeechRecognition);
}

export function listenOnce(onText: (t: string) => void, onEnd: () => void): () => void {
  const w = window as unknown as Record<string, unknown>;
  const Ctor = (w.SpeechRecognition || w.webkitSpeechRecognition) as
    | (new () => {
        continuous: boolean;
        interimResults: boolean;
        lang: string;
        start: () => void;
        stop: () => void;
        onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
        onend: (() => void) | null;
      })
    | undefined;
  if (!Ctor) {
    onEnd();
    return () => {};
  }
  const rec = new Ctor();
  rec.continuous = false;
  rec.interimResults = false;
  rec.lang = "en-US";
  rec.onresult = (e) => {
    let out = "";
    for (let i = 0; i < e.results.length; i++) out += e.results[i][0].transcript;
    onText(out.trim());
  };
  rec.onend = onEnd;
  rec.start();
  return () => rec.stop();
}
