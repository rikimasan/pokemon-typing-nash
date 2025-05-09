import itertools as it
import numpy as np
from scipy.optimize import linprog

###############################################################################
# 1.  Data: 18×18 single-type effectiveness matrix (GEN IX)
###############################################################################
types = [
    "Normal","Fire","Water","Electric","Grass","Ice","Fighting","Poison","Ground",
    "Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark","Steel","Fairy"
]

# effectiveness[a][d]  =  damage multiplier of an a-type move on a d-type defender
# source: official type chart – values: 0 , 0.5 , 1 , 2
E = np.array([
# Nor Fir Wat Ele Gra Ice Fig Poi Gro Fly Psy Bug Roc Gho Dra Dar Ste Fai
 [1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ,0.5, 0 , 1 , 1 ,0.5, 1 ], # Normal
 [1 ,0.5,0.5, 1 ,2 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,0.5,1 ,0.5,1 ,2 ,1 ],             # Fire
 [1 ,2 ,0.5,1 ,0.5,1 ,1 ,1 ,2 ,1 ,1 ,1 ,2 ,1 ,0.5,1 ,1 ,1 ],              # Water
 [1 ,1 ,2 ,0.5,0.5,1 ,1 ,1 ,0 ,2 ,1 ,1 ,1 ,1 ,0.5,1 ,1 ,1 ],              # Electric
 [1 ,0.5,2 ,1 ,0.5,1 ,1 ,0.5,2 ,0.5,1 ,0.5,2 ,1 ,0.5,1 ,0.5,1 ],           # Grass
 [1 ,0.5,0.5,1 ,2 ,0.5,1 ,1 ,2 ,2 ,1 ,1 ,1 ,1 ,2 ,1 ,0.5,1 ],             # Ice
 [2 ,1 ,1 ,1 ,1 ,2 ,1 ,0.5,1 ,0.5,0.5,0.5,2 ,0 ,1 ,2 ,2 ,0.5],            # Fighting
 [1 ,1 ,1 ,1 ,2 ,1 ,1 ,0.5,0.5,1 ,1 ,1 ,0.5,0.5,1 ,1 ,0 ,2 ],            # Poison
 [1 ,2 ,1 ,2 ,0.5,1 ,1 ,2 ,1 ,0 ,1 ,0.5,2 ,1 ,1 ,1 ,2 ,1 ],              # Ground
 [1 ,1 ,1 ,0.5,2 ,1 ,2 ,1 ,1 ,1 ,1 ,2 ,0.5,1 ,1 ,1 ,0.5,1 ],             # Flying
 [1 ,1 ,1 ,1 ,1 ,1 ,2 ,2 ,1 ,1 ,0.5,1 ,1 ,1 ,1 ,0 ,0.5,1 ],              # Psychic
 [1 ,0.5,1 ,1 ,2 ,1 ,0.5,0.5,1 ,0.5,2 ,1 ,1 ,0.5,1 ,2 ,0.5,0.5],         # Bug
 [1 ,2 ,1 ,1 ,1 ,2 ,0.5,1 ,0.5,2 ,1 ,2 ,1 ,1 ,1 ,1 ,0.5,1 ],             # Rock
 [0 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,0.5,1 ,1 ],              # Ghost
 [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,0.5,0 ],              # Dragon
 [1 ,1 ,1 ,1 ,1 ,1 ,0.5,1 ,1 ,1 ,2 ,1 ,1 ,2 ,1 ,0.5,1 ,0.5],            # Dark
 [1 ,0.5,0.5,0.5,1 ,2 ,1 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,0.5,2 ],           # Steel
 [1 ,0.5,1 ,1 ,1 ,1 ,2 ,0.5,1 ,1 ,1 ,1 ,1 ,1 ,2 ,2 ,0.5,1 ],            # Fairy
])

###############################################################################
# 2.  Enumerate single & dual type combinations
###############################################################################
singles = [(t,) for t in range(18)]
duals   = [c for c in it.combinations(range(18), 2)]
combos  = singles + duals           # total 171

def effectiveness(attack_type: int, defender_combo: tuple[int]) -> float:
    """Return damage multiplier of 'attack_type' vs defender with 1 or 2 types."""
    mult = 1.
    for d in defender_combo:
        mult *= E[attack_type, d]
    return mult

###############################################################################
# 3.  Build payoff matrix  P[i, j]  for row-player combo i vs column-player combo j
#     Score rule (blog): winner keeps  1 / dmg_ratio  of its HP.
###############################################################################
N = len(combos)
P = np.zeros((N, N))

for i, atk in enumerate(combos):
    # for attack combo, we can choose whichever move (its own type(s)) deals more damage
    atk_moves = atk                      # 1 or 2 integers
    for j, def_ in enumerate(combos):
        # best damage we (row) can deal to defender
        best = max(effectiveness(m, def_) for m in atk_moves)

        # Column gets its best retaliation damage
        best_ret = max(
            effectiveness(m, atk) for m in def_
        )

        # If both best damages equal ⇒ draw ⇒ score 0
        if np.isclose(best, best_ret):
            s_row = s_col = 0.
        elif best > best_ret:
            # row wins, keeps fraction = 1 - best_ret / best
            s_row = 1 - best_ret / best
            s_col = -s_row
        else:
            s_col = 1 - best / best_ret
            s_row = -s_col

        P[i, j] = s_row

###############################################################################
# 4.  Solve for a mixed strategy of the row player via a single linear program
#
#     Maximise  v
#     subject to   P^T p ≥ v·1
#                  1^T p = 1 ,  p ≥ 0
#
#     Transform to standard form for scipy.optimize.linprog (minimisation):
#     minimise -v    with variables [p_0 … p_{N-1} , v]
###############################################################################
c         = np.r_[np.zeros(N), -1]        # minimise −v  ⇒  maximise v
A_ub      = np.c_[ -P.T ,  np.ones((N,1))]  # -P^T p + v·1 ≤ 0
b_ub      = np.zeros(N)
A_eq = np.hstack([np.ones((1, N)), np.zeros((1, 1))])          # (1, N+1)
b_eq      = np.array([1.])

bounds    = [(0, None)]*N + [(None, None)]   # p_i ≥ 0 , v free
res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds, method="highs")

if not res.success:
    raise RuntimeError(res.message)

p_star = res.x[:-1]          # equilibrium probabilities
v_star = res.x[-1]

###############################################################################
# 5.  Produce the blog’s “Nash score” for each combo   (probability * value)
###############################################################################
nash_scores = p_star * 1        # identical to probability weights because zero-sum
order = nash_scores.argsort()[::-1]
nonzero = order[nash_scores[order] > 1e-12]

print("Top combinations (probability ≈ Nash score):")
for k, idx in enumerate(nonzero, 1):
    combo = "/".join(types[t] for t in combos[idx])
    print(f"{k:2}. {combo:<12}  {nash_scores[idx]:.4f}")

###############################################################################
# 6.  Aggregate by single types – sum of probabilities of every combo containing it
###############################################################################
type_sum = np.zeros(18)
for i, prob in enumerate(p_star):
    for t in combos[i]:
        type_sum[t] += prob

print("\nAggregate per single type:")
for idx in type_sum.argsort()[::-1]:
    print(f"{types[idx]:<8} {type_sum[idx]:.4f}")
