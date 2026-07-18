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

# Published-style PdG confirmation: a sustained rise over the days AFTER a candidate ovulation,
# relative to the days BEFORE it, using an absolute-fraction threshold rather than a within-window
# SD (which would be circular). Threshold is on normalized PdG within the cycle.
PDG_RISE_FRACTION = 0.25


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
    """Return (ovulation_day, anovulatory) for one cycle slice using LH argmax + PdG confirmation.

    Used for real data where no ground-truth ovulation column exists. The PdG check compares the
    normalized post-candidate mean against the pre-candidate mean; a rise below PDG_RISE_FRACTION
    is treated as unconfirmed (anovulatory).
    """
    g = cycle_slice.sort_values(C.KEY_DAY).reset_index(drop=True)
    if g["lh"].isna().all():
        return -1, True
    ov_idx = int(g["lh"].to_numpy().argmax())
    pdg = g["pdg"].to_numpy(dtype=float)
    if np.all(np.isnan(pdg)) or np.nanmax(pdg) <= np.nanmin(pdg):
        return -1, True
    pdg_norm = (pdg - np.nanmin(pdg)) / (np.nanmax(pdg) - np.nanmin(pdg) + 1e-9)
    pre = pdg_norm[max(0, ov_idx - 6) : ov_idx]
    post = pdg_norm[ov_idx + 1 : ov_idx + 6]
    if len(pre) == 0 or len(post) == 0:
        return -1, True
    rise = float(np.nanmean(post) - np.nanmean(pre))
    if rise < PDG_RISE_FRACTION:
        return int(g[C.KEY_DAY].iloc[ov_idx]), True  # LH peak location kept, but flagged anovulatory
    return int(g[C.KEY_DAY].iloc[ov_idx]), False


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
