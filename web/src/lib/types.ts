// Types mirroring the Python export (scripts/export_web_data.py). Kept in one place so the
// frontend and the export stay in sync.

export type Regularity = "regular" | "irregular";
export type HormoneKey = "e3g" | "pdg" | "lh";

export interface Band {
  mean: (number | null)[];
  lo: (number | null)[];
  hi: (number | null)[];
}

export interface CycleSpan {
  index: number;
  startDay: number;
  endDay: number;
  lengthDays: number;
  anovulatory: boolean;
}

export interface OvulationEvent {
  truthDay: number;
  calendarDay: number;
  modelDay: number;
}

export interface Participant {
  pid: string;
  regularity: Regularity;
  days: number[];
  truth: Record<HormoneKey, (number | null)[]>;
  model: Record<HormoneKey, Band>;
  cycles: CycleSpan[];
  events: OvulationEvent[];
}

export interface ParticipantIndex {
  pid: string;
  regularity: Regularity;
  nCycles: number;
  cycleLenStd: number | null;
}

export interface SkillCell {
  soc: number | null;
  lo: number | null;
  hi: number | null;
  n: number;
  p: number | null;
  primary: boolean;
}

export interface SkillStratum {
  stratum: string;
  synthetic: SkillCell;
  real: SkillCell;
}

export interface LeaderboardRow {
  task: string;
  label: string;
  metric: string;
  synthetic: number | null;
  real: number | null;
  transfer: number | null;
  n_real: number;
}
