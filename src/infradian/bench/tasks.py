"""Task construction: extract cycles and ground-truth ovulation labels from the canonical frame.

A cycle spans one observed menses onset to the next. Ground-truth ovulation:
  - Synthetic (Tier C): read the generator's `_ovulation_day` (and `_anovulatory`) metadata.
  - Real (Tier B): argmax of LH within the cycle, confirmed by a sustained PdG rise (see
    `confirm_ovulation`). Cycles failing confirmation are labelled anovulatory (task T2-A) and
    excluded from T2-R/T2-P day-localization scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from infradian.bench.splits import regularity_of
from infradian.data import canonical as C

# Ovulation is anchored on the LH SURGE (the clinical gold-standard predictor; ovulation follows
# the surge by ~24h). PdG provides corroboration WHEN available, but must not veto a clear surge
# just because PdG was not sampled in the pre-ovulation window — in mcPHASES, PdG is measured
# selectively (mostly in the luteal phase), so a PdG-mandatory rule would spuriously flag nearly
# every cycle anovulatory. A cycle is anovulatory only when no clear LH surge is present.
PDG_RISE_FRACTION = 0.20
LH_SURGE_ABS = 15.0  # absolute LH floor for a surge (Mira-scale; baseline LH ~1-5)
LH_SURGE_REL = 2.5  # surge must exceed this multiple of the cycle's median LH


@dataclass
class Cycle:
    participant_id: str
    segment_id: str
    cycle_start: int  # day_in_study of the onset
    cycle_end: int  # day_in_study of the next onset (exclusive) or segment end
    cycle_len: int
    true_ov_day: int  # day_in_study of ovulation, or -1 if anovulatory/unconfirmed
    anovulatory: bool
    regularity: str


def _onset_days(g: pd.DataFrame) -> list[int]:
    m = g["menses_reported"].fillna(0).to_numpy().astype(int)
    days = g[C.KEY_DAY].to_numpy()
    return [int(days[i]) for i in range(len(m)) if m[i] == 1 and (i == 0 or m[i - 1] == 0)]


def confirm_ovulation(cycle_slice: pd.DataFrame) -> tuple[int, bool]:
    """Return (ovulation_day, anovulatory) for one cycle slice.

    Rule (robust to PdG sparsity):
      1. Ovulation day = day of the LH peak.
      2. The cycle is OVULATORY if the LH peak is a clear surge: above an absolute floor AND a
         multiple of the cycle's median LH.
      3. PdG, when sampled around the candidate day, must not CONTRADICT it — a clearly falling PdG
         after a putative surge downgrades to anovulatory. Absent PdG does not veto.
    """
    g = cycle_slice.sort_values(C.KEY_DAY).reset_index(drop=True)
    lh = g["lh"].to_numpy(dtype=float)
    if np.all(np.isnan(lh)):
        return -1, True
    ov_idx = int(np.nanargmax(lh))
    ov_day = int(g[C.KEY_DAY].iloc[ov_idx])
    peak = float(lh[ov_idx])
    med = float(np.nanmedian(lh))

    is_surge = (peak >= LH_SURGE_ABS) and (peak >= LH_SURGE_REL * max(med, 1e-6))
    if not is_surge:
        return ov_day, True  # no clear surge -> anovulatory

    # Optional PdG corroboration: only downgrade if PdG data exists on BOTH sides and clearly falls.
    pdg = g["pdg"].to_numpy(dtype=float)
    pre = pdg[max(0, ov_idx - 6) : ov_idx]
    post = pdg[ov_idx + 1 : ov_idx + 7]
    if np.isfinite(pre).sum() >= 2 and np.isfinite(post).sum() >= 2:
        rise = float(np.nanmean(post) - np.nanmean(pre))
        span = float(np.nanmax(pdg) - np.nanmin(pdg)) if np.isfinite(pdg).any() else 0.0
        if span > 0 and rise < -PDG_RISE_FRACTION * span:  # PdG clearly falls -> contradicts surge
            return ov_day, True
    return ov_day, False


def extract_cycles(df: pd.DataFrame, use_metadata: bool = True) -> list[Cycle]:
    """Extract cycles per segment. If `use_metadata` and `_ovulation_day` is present (synthetic),
    read ground truth from it; otherwise confirm ovulation from LH/PdG (real data)."""
    cycles: list[Cycle] = []
    reg_by_pid = {}
    for pid, gp in df.groupby(C.KEY_PARTICIPANT):
        onsets_all = _onset_days(gp.sort_values(C.KEY_DAY))
        lens = np.diff(onsets_all) if len(onsets_all) >= 2 else np.array([])
        reg_by_pid[pid] = regularity_of(lens)

    has_meta = use_metadata and "_ovulation_day" in df.columns
    for seg, g in df.groupby(C.KEY_SEGMENT, sort=False):
        g = g.sort_values(C.KEY_DAY).reset_index(drop=True)
        pid = g[C.KEY_PARTICIPANT].iloc[0]
        onsets = _onset_days(g)
        seg_end = int(g[C.KEY_DAY].iloc[-1]) + 1
        for k, start in enumerate(onsets):
            end = onsets[k + 1] if k + 1 < len(onsets) else seg_end
            if end - start < 15:  # too short to be a real closed cycle
                continue
            cyc = g[(g[C.KEY_DAY] >= start) & (g[C.KEY_DAY] < end)]
            if has_meta:
                ovs = cyc["_ovulation_day"].to_numpy()
                ov_vals = ovs[ovs >= 0]
                if len(ov_vals) > 0:
                    true_ov, anov = int(ov_vals[0]), False
                else:
                    true_ov, anov = -1, True
            else:
                true_ov, anov = confirm_ovulation(cyc)
            cycles.append(
                Cycle(
                    participant_id=pid,
                    segment_id=seg,
                    cycle_start=start,
                    cycle_end=end,
                    cycle_len=end - start,
                    true_ov_day=true_ov,
                    anovulatory=anov,
                    regularity=reg_by_pid[pid],
                )
            )
    return cycles


def cycles_to_frame(cycles: list[Cycle]) -> pd.DataFrame:
    return pd.DataFrame([c.__dict__ for c in cycles])
