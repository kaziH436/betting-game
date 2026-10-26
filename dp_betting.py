#!/usr/bin/env python3
"""
Optimal betting by dynamic programming.

Game
  - Each bet wins with probability 0.4 and loses with probability 0.6.
  - You start with START chips. The goal is to have placed bets TOTALING at
    least TARGET chips (win or lose, every bet counts its full size).
  - Min bet 1, max bet = chips you currently hold. Hit 0 chips first -> you lose.

State
  x = chips you hold now
  r = wager still needed  (r = TARGET - total wagered so far)

Value
  V[r][x] = best possible probability of winning from state (x, r)

Base cases
  x >= r  ->  V = 1   (bet r chips: it counts fully toward the target, win or lose)
  x == 0  ->  V = 0

Recurrence (bet b, 1 <= b <= x, b < r)
  V[r][x] = max over b of   0.4 * V[r-b][x+b]  +  0.6 * V[r-b][x-b]

Every bet lowers r by at least 1, so V[r] only depends on rows with smaller r.
We fill the table (a 2D array) in increasing r.

Usage examples
  python optimal_betting_dp.py                       # exact, START=200 TARGET=1000 (fast)
  python optimal_betting_dp.py --start 400 --target 2000
  python optimal_betting_dp.py --start 2000 --target 10000 --grid-growth 1.06
        (full size: restricted bet sizes, needs ~600 MB RAM, takes a few minutes)
"""

import argparse
import random
import time

import numpy as np

P_WIN = 0.4
P_LOSE = 0.6


def candidate_bets(target, growth, linear_upto=30):
    """Bet sizes to try. growth == 0 -> every size 1..target-1 (exact)."""
    if growth <= 0:
        return list(range(1, target))
    bets = list(range(1, linear_upto + 1))
    b = float(linear_upto)
    while b < target:
        b *= growth
        if int(b) > bets[-1]:
            bets.append(int(b))
    return [b for b in bets if b < target]


def solve(target, bets):
    """Return V[r, x] and best-bet table B[r, x] (both indexed r=0..T, x=0..T)."""
    T = target
    V = np.zeros((T + 1, T + 1), dtype=np.float32)
    B = np.zeros((T + 1, T + 1), dtype=np.int16)

    V[0, :] = 1.0
    for r in range(1, T + 1):
        V[r, r:] = 1.0        # x >= r : guaranteed win
        B[r, r:] = r          # ...by betting r
        # V[r, 0] stays 0 (out of chips)

    for r in range(1, T + 1):
        if r == 1:
            continue
        xs = np.arange(1, r) # states with 1 <= x < r
        best = np.full(r - 1, -1.0, dtype=np.float32)
        best_bet = np.ones(r - 1, dtype=np.int32)

        # Fixed-size candidate bets (need b <= x)
        for b in bets:
            if b >= r:
                break
            sel = xs >= b
            x = xs[sel]
            win_state = V[r - b, np.minimum(x + b, T)]   # clipped: x' >= r' means value 1
            lose_state = V[r - b, x - b]
            val = P_WIN * win_state + P_LOSE * lose_state
            idx = np.where(sel)[0]
            better = val > best[idx] + 1e-9 # ties keep the smaller bet
            best[idx[better]] = val[better]
            best_bet[idx[better]] = b

        # All-in candidate (b = x), always considered
        win_state = V[r - xs, np.minimum(2 * xs, T)]
        val = P_WIN * win_state  # lose -> 0 chips -> value 0
        better = val > best + 1e-9
        best[better] = val[better]
        best_bet[better] = xs[better]

        V[r, 1:r] = best
        B[r, 1:r] = best_bet
    return V, B


def print_policy_snapshot(V, B, start, target):
    print("\n=== OPTIMAL POLICY SNAPSHOT ===")
    print("line = 0.2 * r  (chips you expect to lose over the remaining wager)\n")
    factors = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    for frac in (1.0, 0.8, 0.5, 0.2):
        r = int(round(target * frac))
        line = 0.2 * r
        print(f"r = {r}  (wagered so far W = {target - r}),  line = {line:.0f}")
        print(f"   {'chips x':>9} {'x/line':>7} {'best bet':>9} {'V(x,r)':>9}")
        for f in factors:
            x = int(round(f * line))
            if x < 1 or x >= r:
                continue
            print(f"   {x:>9} {f:>7.2f} {int(B[r, x]):>9} {float(V[r, x]):>9.4f}")
        print()


def play_one_game(B, start, target, rng):
    """Follow the optimal policy, return (won, rows). Each row describes one bet."""
    x, wagered = start, 0
    rows = []
    n = 0
    while wagered < target and x > 0:
        r = target - wagered
        b = r if x >= r else int(B[r, x])
        n += 1
        win = rng.random() < P_WIN
        x_after = x + b if win else x - b
        rows.append((n, wagered, r, x, b, win, x_after))
        wagered += b
        x = x_after
    return wagered >= target, rows


def print_trace(V, rows, target, head, every, tail):
    print("\n=== SAMPLE GAME FOLLOWING THE OPTIMAL STRATEGY ===")
    print(f"{'bet#':>5} {'wagered':>8} {'r left':>7} {'chips':>7} {'line':>7} "
          f"{'V(x,r)':>8} {'bet':>6} {'result':>7} {'chips after':>12}")
    total = len(rows)
    shown_gap = False
    for i, (n, w, r, x, b, win, xa) in enumerate(rows):
        show = i < head or i >= total - tail or (i % every == 0) or b > 1
        if not show:
            shown_gap = True
            continue
        if shown_gap:
            print("  ...")
            shown_gap = False
        vv = 1.0 if x >= r else float(V[r, x])
        note = "  <- guaranteed finish" if x >= r else ""
        print(f"{n:>5} {w:>8} {r:>7} {x:>7} {0.2 * r:>7.0f} {vv:>8.4f} {b:>6} "
              f"{'WIN' if win else 'LOSE':>7} {xa:>12}{note}")
    print(f"(total bets placed: {total}; rows with bet size > 1 are always shown)")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--grid-growth", type=float, default=0.0,
                    help="0 = try every bet size (exact). e.g. 1.06 = restricted grid (faster)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sims", type=int, default=1000, help="Monte Carlo games to verify V")
    ap.add_argument("--trace-head", type=int, default=15)
    ap.add_argument("--trace-every", type=int, default=100)
    ap.add_argument("--trace-tail", type=int, default=10)
    a = ap.parse_args()

    bets = candidate_bets(a.target, a.grid_growth)
    mode = "exact (all bet sizes)" if a.grid_growth <= 0 else f"restricted grid ({len(bets)} sizes + all-in)"
    print(f"Chips start = {a.start}, wager target = {a.target}, P(win bet) = {P_WIN}")
    print(f"Search mode: {mode}")

    t0 = time.time()
    V, B = solve(a.target, bets)
    print(f"DP finished in {time.time() - t0:.1f}s")

    print("\n=== ANSWER ===")
    print(f"V(x={a.start}, r={a.target}) = {float(V[a.target, a.start]):.4f}   "
          f"<- optimal probability of reaching the wager target")
    print(f"Optimal opening bet = {int(B[a.target, a.start])}")

    print_policy_snapshot(V, B, a.start, a.target)

    rng = random.Random(a.seed)
    won, rows = play_one_game(B, a.start, a.target, rng)
    print_trace(V, rows, a.target, a.trace_head, a.trace_every, a.trace_tail)
    print("Outcome of this sample game:", "REACHED TARGET (win)" if won else "RAN OUT OF CHIPS (loss)")

    if a.sims > 0:
        wins = sum(play_one_game(B, a.start, a.target, rng)[0] for _ in range(a.sims))
        print(f"\n=== MONTE CARLO CHECK ({a.sims} games) ===")
        print(f"simulated win rate = {wins / a.sims:.4f}   vs   DP value = {float(V[a.target, a.start]):.4f}")


if __name__ == "__main__":
    main()