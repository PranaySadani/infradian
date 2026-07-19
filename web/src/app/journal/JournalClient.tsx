"use client";

import { useEffect, useRef, useState } from "react";
import {
  ai,
  blobToBase64,
  browserSpeechAvailable,
  downscaleImage,
  listenOnce,
  type AiStatus,
  type JournalResult,
  type StripReading,
} from "@/lib/ai";

interface Entry {
  id: number;
  text: string;
  source: string;
  symptoms: JournalResult["symptoms"];
  schema: Record<string, number>;
}

interface Anchor {
  id: number;
  analyte: string;
  value: number;
  unit: string;
  confidence: number;
  manual: boolean;
}

export function JournalClient() {
  const [status, setStatus] = useState<AiStatus | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [refusal, setRefusal] = useState<string | null>(null);
  const [listening, setListening] = useState(false);
  const [recording, setRecording] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const [anchors, setAnchors] = useState<Anchor[]>([]);
  const [reading, setReading] = useState<StripReading | null>(null);
  const [imgBusy, setImgBusy] = useState(false);
  const [manual, setManual] = useState({ analyte: "LH", value: "" });

  const stopRef = useRef<() => void>(() => {});
  const recRef = useRef<MediaRecorder | null>(null);

  useEffect(() => {
    ai.status().then(setStatus);
  }, []);

  const submit = async (raw?: string) => {
    const t = (raw ?? text).trim();
    if (!t) return;
    setBusy(true);
    setRefusal(null);
    setNote(null);
    try {
      const r = await ai.journal(t);
      if (r.refused) {
        setRefusal(r.message ?? "That question cannot be answered here.");
        return;
      }
      setEntries((p) => [
        { id: Date.now(), text: r.text ?? t, source: r.source ?? "deterministic", symptoms: r.symptoms, schema: r.schema_fields },
        ...p,
      ]);
      setText("");
      if (r.notes?.length) setNote(r.notes[0]);
    } catch {
      setNote("Could not reach the extraction service. Your entry was not lost, try again.");
    } finally {
      setBusy(false);
    }
  };

  // Voice. OpenAI Whisper when a key is configured, browser speech recognition otherwise.
  const startVoice = async () => {
    if (status?.features.voice_transcription) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const rec = new MediaRecorder(stream);
        const chunks: Blob[] = [];
        rec.ondataavailable = (e) => chunks.push(e.data);
        rec.onstop = async () => {
          stream.getTracks().forEach((t) => t.stop());
          setRecording(false);
          setBusy(true);
          try {
            const b64 = await blobToBase64(new Blob(chunks, { type: "audio/webm" }));
            const r = await ai.transcribe(b64);
            if (r.available && r.text) {
              setText(r.text);
              await submit(r.text);
            } else {
              setNote(r.message ?? "Transcription unavailable.");
            }
          } catch {
            setNote("Transcription failed. Type the entry instead.");
          } finally {
            setBusy(false);
          }
        };
        recRef.current = rec;
        rec.start();
        setRecording(true);
        setTimeout(() => rec.state === "recording" && rec.stop(), 20000);
      } catch {
        setNote("Microphone permission denied.");
      }
      return;
    }

    if (browserSpeechAvailable()) {
      setListening(true);
      stopRef.current = listenOnce(
        (t) => {
          setText(t);
          submit(t);
        },
        () => setListening(false)
      );
      return;
    }
    setNote("Voice needs either an OpenAI key or a browser with speech recognition. Type instead.");
  };

  const onPhoto = async (file: File) => {
    setImgBusy(true);
    setReading(null);
    try {
      const dataUri = await downscaleImage(file);
      const r = await ai.readStrip(dataUri);
      setReading(r);
      if (r.readable && r.analyte && r.value != null) {
        setAnchors((p) => [
          { id: Date.now(), analyte: r.analyte!, value: r.value!, unit: r.unit ?? "", confidence: r.confidence, manual: false },
          ...p,
        ]);
      }
    } catch {
      setReading(null);
      setNote("Could not read that image. Enter the value by hand below.");
    } finally {
      setImgBusy(false);
    }
  };

  const addManual = () => {
    const v = parseFloat(manual.value);
    if (!Number.isFinite(v)) return;
    const units: Record<string, string> = { LH: "mIU/mL", E3G: "ng/mL", PdG: "ug/mL", FSH: "mIU/mL" };
    setAnchors((p) => [
      { id: Date.now(), analyte: manual.analyte, value: v, unit: units[manual.analyte] ?? "", confidence: 1, manual: true },
      ...p,
    ]);
    setManual({ ...manual, value: "" });
  };

  const aiOn = status?.enabled ?? false;

  return (
    <div className="max-w-shell mx-auto px-6 py-12">
      <div className="flex items-baseline justify-between flex-wrap gap-3 mb-8">
        <div>
          <p className="eyebrow mb-2">Journal</p>
          <h1 className="heading text-[30px]">Log how you feel. In your own words.</h1>
        </div>
        <span
          className={`px-3 py-1.5 rounded-pill text-[12px] border ${
            aiOn ? "bg-accent-dim text-accent border-[rgba(217,238,80,0.4)]" : "border-hair text-ink-muted"
          }`}
          title={status?.note}
        >
          {aiOn ? `AI on · ${status?.chat_model}` : "AI off · deterministic fallbacks"}
        </span>
      </div>

      <p className="text-[15px] text-ink-secondary leading-[1.65] max-w-[70ch] mb-8">
        Symptom logs are the cheapest longitudinal signal in women&apos;s health and the least
        comparable, because every app invents its own labels. This normalises free text into{" "}
        <span className="text-ink">INFRADIAN-SYM</span>, an open vocabulary, and maps each code onto
        the exact schema columns the model was trained on. Speak it, type it, or photograph a test.
      </p>

      <div className="grid lg:grid-cols-2 gap-4">
        {/* --- log --- */}
        <div className="card p-6">
          <p className="eyebrow mb-4">1 · Describe your day</p>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={4}
            placeholder="Really bad cramps today and I barely slept. Feeling low and bloated."
            className="w-full bg-raised border border-hair rounded-xl p-4 text-[15px] leading-relaxed outline-none focus:border-strong resize-none"
          />
          <div className="flex flex-wrap gap-2 mt-3">
            <button onClick={() => submit()} disabled={busy || !text.trim()} className="pill pill-accent disabled:opacity-40">
              {busy ? "Reading…" : "Extract symptoms"}
            </button>
            <button
              onClick={recording ? () => recRef.current?.stop() : listening ? () => stopRef.current() : startVoice}
              disabled={busy}
              className={`pill ${recording || listening ? "pill-on" : "pill-ghost"} disabled:opacity-40`}
            >
              {recording ? "◼ Stop recording" : listening ? "◼ Listening…" : "◉ Speak instead"}
            </button>
          </div>
          <p className="text-[12px] text-ink-muted mt-3 leading-relaxed">
            {aiOn
              ? "Voice goes to Whisper, then the text is normalised by the model against the fixed vocabulary."
              : browserSpeechAvailable()
              ? "No key configured, so voice uses your browser's own speech recognition and the deterministic extractor."
              : "No key configured. Typing uses the deterministic extractor, which needs no network."}
          </p>

          {refusal && (
            <p className="mt-4 text-[14px] text-warning leading-relaxed border-l-2 border-warning pl-4">{refusal}</p>
          )}
          {note && <p className="mt-3 text-[12px] text-ink-muted">{note}</p>}
        </div>

        {/* --- anchor --- */}
        <div className="card p-6">
          <p className="eyebrow mb-4">2 · Anchor it with a real measurement</p>
          <p className="text-[14px] text-ink-secondary leading-[1.6] mb-4">
            A reconstructed curve with no anchor is a guess with a nice shape. Photograph an at-home
            LH or hormone test and the reading becomes a fixed point on your trajectory.
          </p>
          <label className={`pill ${imgBusy ? "pill-on" : "pill-ghost"} inline-block cursor-pointer`}>
            {imgBusy ? "Reading photo…" : "◧ Photograph a test"}
            <input
              type="file"
              accept="image/*"
              capture="environment"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && onPhoto(e.target.files[0])}
            />
          </label>

          {reading && (
            <div className="mt-4 rounded-xl bg-raised border border-hair p-4">
              {reading.readable ? (
                <>
                  <div className="num text-[24px] text-accent">
                    {reading.analyte} {reading.value} <span className="text-[14px] text-ink-muted">{reading.unit}</span>
                  </div>
                  <div className="text-[12px] text-ink-muted mt-1">confidence {reading.confidence}</div>
                </>
              ) : (
                <div className="text-[13px] text-ink-secondary">{reading.reason}</div>
              )}
              <p className="text-[11px] text-ink-muted mt-3 leading-relaxed">{reading.disclaimer}</p>
              {reading.warnings.map((w) => (
                <p key={w} className="text-[11px] text-warning mt-1">{w}</p>
              ))}
            </div>
          )}

          <div className="mt-5 pt-5 border-t border-hair">
            <p className="eyebrow mb-3">or enter it by hand</p>
            <div className="flex gap-2">
              <select
                value={manual.analyte}
                onChange={(e) => setManual({ ...manual, analyte: e.target.value })}
                className="pill pill-ghost appearance-none cursor-pointer"
              >
                {["LH", "E3G", "PdG", "FSH"].map((a) => (
                  <option key={a} value={a} className="bg-raised">{a}</option>
                ))}
              </select>
              <input
                value={manual.value}
                onChange={(e) => setManual({ ...manual, value: e.target.value })}
                placeholder="value"
                inputMode="decimal"
                className="pill pill-ghost w-28 num outline-none"
              />
              <button onClick={addManual} className="pill pill-ghost">Add</button>
            </div>
          </div>

          {anchors.length > 0 && (
            <div className="mt-5 flex flex-wrap gap-2">
              {anchors.map((a) => (
                <span key={a.id} className="px-3 py-1.5 rounded-pill border border-hair text-[12px] num text-ink-secondary">
                  {a.analyte} {a.value} {a.unit}
                  <span className="text-ink-muted"> · {a.manual ? "manual" : "photo"}</span>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* --- extracted --- */}
      {entries.length > 0 && (
        <div className="card p-6 mt-4">
          <div className="flex items-center justify-between mb-5 flex-wrap gap-2">
            <p className="eyebrow">3 · Normalised into the open vocabulary</p>
            <span className="text-[12px] text-ink-muted num">INFRADIAN-SYM v1.0</span>
          </div>
          <div className="space-y-5">
            {entries.map((e) => (
              <div key={e.id} className="border-b border-hair last:border-0 pb-5 last:pb-0">
                <p className="text-[15px] text-ink-secondary leading-relaxed">&ldquo;{e.text}&rdquo;</p>
                <div className="flex flex-wrap gap-1.5 mt-3">
                  {e.symptoms.map((s) => (
                    <span
                      key={s.code}
                      title={`${s.label} · severity ${s.severity}/4${s.evidence ? ` · "${s.evidence}"` : ""}`}
                      className="px-2.5 py-1 rounded-pill border border-hair text-[11px] num text-ink-muted"
                    >
                      {s.code.replace("SYM.", "")} <span className="text-accent">{s.severity}</span>
                    </span>
                  ))}
                  {e.symptoms.length === 0 && (
                    <span className="text-[12px] text-ink-muted">no known symptom terms matched</span>
                  )}
                </div>
                {Object.keys(e.schema).length > 0 && (
                  <p className="text-[12px] text-ink-muted mt-3 num">
                    feeds model schema: {Object.entries(e.schema).map(([k, v]) => `${k}=${v}`).join("  ")}
                  </p>
                )}
                <p className="text-[11px] text-ink-muted mt-2">
                  extracted by {e.source === "openai" ? "OpenAI, validated against the vocabulary" : "the deterministic matcher"}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="text-[12px] text-ink-muted mt-6 max-w-[80ch] leading-relaxed">
        Nothing here is stored on a server. Entries live in this browser tab only. This is a research
        tool, not a diagnostic device, and it will refuse to answer diagnostic questions.
      </p>
    </div>
  );
}
