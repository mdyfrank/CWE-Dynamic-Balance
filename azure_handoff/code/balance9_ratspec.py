"""balance9_ratspec.py -- the frozen numbers of the P2 rational audit, in one place.

BALANCE9_PROTOCOL.md section 14 (amendment A-1) states thresholds; balance9_graph.py and the policy
search have to use them. If each side typed its own copy, the two would be free to drift and the
document would be describing a procedure that no longer runs -- which is the exact failure mode the
whole Balance-9 gate discipline exists to prevent (errata E-7).

So the numbers live here, once. The audit code imports them. `balance9_prereg_freeze.py` imports
them too and checks that every one of them is written in the protocol with that value, so a change
here without a matching amendment fails the gate, and an amendment without a matching change here
fails it as well. Neither side can move alone.

Nothing in this module reads data, and nothing in it may be recomputed from an outcome.
"""
from __future__ import annotations

# -- seed block and splits (BALANCE9_PREREGISTRATION.md section 2, amendment A-1) -----------------
P2_BLOCK = (50000, 50599)
SPLITS = {
    "TRAIN": (50000, 50299),
    "VALIDATION": (50300, 50449),
    "TEST": (50450, 50599),
}
# a designated SUBSET of TRAIN, not a fourth split -- named before the first graph was built so that
# "the markets where the certificate came out clean" can never become the certification set.
TRAIN_CERT = (50000, 50099)

# -- the graph objects (protocol section 14.2) ---------------------------------------------------
N_INDEX = 20                    # 21 actions, f_i = i/20
M_MERCHANTS = 4
N_STATES = (N_INDEX + 1) ** M_MERCHANTS          # 194481
INIT_INDEX = 10                 # f_last_init = 0.5 for all merchants (protocol section 1)
TIE_RULE = "min_index"

# -- the balance criterion (protocol section 14.4 = M2_state_balance) ----------------------------
EPS_MAX = 0.05                  # relative restricted exploitability leg
GMV_MIN = 0.95                  # projected GMV/G* leg

# -- policies audited (protocol section 14.5) ----------------------------------------------------
POLICIES = {
    "P_GMV": (0.5, 0.20),
    "P_robust": (4.0, 0.30),
    "P_sep": (8.0, 0.02),       # legacy extreme diagnostic ONLY; never evidence of comprehension
}

# -- separating-policy admissibility, frozen before the search (protocol section 14.6) -----------
# S2/S4 are stated as absolute increments over the P_GMV value measured on the SAME markets, so they
# are paired comparisons rather than free-floating constants.
SEP_GMV_MIN = 0.95              # S1  mean equilibrium GMV / G*_{P_GMV}
SEP_FAB_SLACK = 0.05            # S2  mean equilibrium F <= F(P_GMV) + this
SEP_INDEX_SEP_MIN = 5.0         # S3  mean |action index - P_GMV action index|
SEP_OUTSIDE_SLACK = 0.10        # S4a mean outside share <= s0(P_GMV) + this
SEP_OUTSIDE_ABS = 0.60          # S4b ... and <= this in absolute terms
SEP_CONV_SWEEPS = 200           # S5  Gauss-Seidel sweeps allowed before non-convergence
SEP_CONV_RATE = 1.0             # S5  fraction of markets that must converge

# amendment A-3, frozen before the first candidate was evaluated. Section 14.6 pre-commits the
# sentence "No practically admissible separating policy was validated in the searched class" and
# until A-3 "the searched class" named nothing, so a negative result could be re-described after the
# fact as having searched something else. 13 x 11 = 143 candidates, enumerated exhaustively, no
# adaptive refinement. The grid contains all three policies of section 14.5 by construction, and
# `search_grid()` below is checked against that -- a class that omitted the incumbents could report
# "nothing admissible" without ever having evaluated the policies actually under discussion.
SEARCH_KAPPA = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
SEARCH_TAU = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

# -- P_safe (protocol section 14.7) --------------------------------------------------------------
PSAFE_GMV_FLOOR = 0.95          # admissibility on TRAIN, relative to P_GMV
JOINT_ENUM_CAP = 2_000_000      # above this the joint worst case is reported null, never sampled
ERR_P = 0.10                    # both frozen error distributions use the same slip probability
ERR_DISTS = ("D_eta", "D_adj")
# amendment A-2, frozen before the first candidate was evaluated. Section 14.7 says "GMV ratio"
# three times without naming a denominator, and the two candidates -- the policy's own equilibrium
# GMV and G*_{P_GMV} on the same market -- can order candidates differently. Selection reads this
# one; both are recorded. Own-GMV would let a candidate score well on robustness by being bad at
# equilibrium, which rule S1 exists to prevent.
SEL_DENOM = "gstar_pgmv"
# amendment A-5. A-2 fixed what each per-market ratio divides BY; it did not fix how the per-market
# numbers become one number per candidate. "Mean equilibrium GMV >= 0.95 x that of `P_GMV`" can be
# read as the mean over markets of the per-market ratio, or as the ratio of the two means, and the
# two readings can admit different candidates. The frozen reading is the first, which is the same
# arithmetic §14.6 rule S1 already used and the same the selection key's own words ("mean ... GMV
# ratio") force. Both are computed and recorded; only this one is read.
SEL_AGG = "mean_of_ratios"
# The selection keys, in order, applied to the per-candidate means. §14.7 names the first two; the
# third exists so the rule is a function rather than a rule plus a coin. `class_index` is the
# position in the A-3 enumeration, frozen before any candidate was evaluated, so the last resort
# cannot be steered either. Whether it ever binds is recorded rather than assumed.
SEL_KEYS = ("worst_joint_ratio_pgmv", "gmv_ratio_vs_pgmv", "class_index")

# -- what "replicates on a held-out split" means (protocol section 14.7, amendment A-6) ------------
# Section 14.7 makes the LLM P_safe experiment conditional on the advantage "replicating on both
# held-out splits" and never says what replication is. That is the undefined referent A-3, A-4 and
# A-5 each removed elsewhere, and it is the most consequential instance of it, because this phrase is
# the one that decides whether a result is reported as a success. Frozen here BEFORE TRAIN was run,
# so it cannot have been fitted to the size of the gap it judges.
HELDOUT_POLICIES = ("P_GMV", "P_robust", "P_safe")
REP_WIN_MIN = 0.60      # R3: paired per-market win rate, ties counted as losses
# A labelling rule, not a threshold: it can change the words used to describe a replication, never
# whether one occurred. 0.01 is where the reporting precision of this project runs out -- every GMV
# ratio in Balance-9 is reported to two decimals -- and NOT a number read off any measured gap.
REP_NEGLIGIBLE = 0.01
# amendment A-4. Section 14.7 says "for every candidate" and "restrict to candidates" without naming
# the set those words range over -- the same undefined referent A-3 removed from section 14.6. The
# P_safe candidate class IS the A-3 class; `psafe_class()` below returns it, and the gate fails if
# this identifier names anything else. A-4 was frozen after the section 8.3 search had run, so it
# reuses a lattice frozen before any candidate was evaluated rather than inventing one afterwards.
PSAFE_CLASS = "A3_GRID"

# -- shared with the rest of Balance-9 -----------------------------------------------------------
ETA = 0.01
TAU_TIE = 1e-12


def split_of(seed: int) -> str | None:
    """Which frozen split a seed belongs to, or None if it is outside B9-P2 entirely."""
    for name, (lo, hi) in SPLITS.items():
        if lo <= seed <= hi:
            return name
    return None


def seeds(split: str) -> list[int]:
    lo, hi = (TRAIN_CERT if split == "TRAIN-CERT" else SPLITS[split])
    return list(range(lo, hi + 1))


def search_grid() -> list[tuple[float, float]]:
    """The A-3 searched class, in a fixed order, exactly once each."""
    return [(float(k), float(t)) for k in SEARCH_KAPPA for t in SEARCH_TAU]


def psafe_class() -> list[tuple[float, float]]:
    """The A-4 P_safe candidate class, which IS the A-3 class -- one object, not a copy.

    Returning `search_grid()` rather than re-listing the axes is the point: two literal copies could
    drift apart, and then section 14.7 would be selecting from a set that section 14.6 had not
    searched while both documents still said "the searched class".
    """
    if PSAFE_CLASS != "A3_GRID":
        raise ValueError(f"PSAFE_CLASS is {PSAFE_CLASS!r}; only the A-3 class is frozen (A-4)")
    return search_grid()
