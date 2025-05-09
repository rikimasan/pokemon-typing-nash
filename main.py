"""
Re-implementation of
“The best Pokémon type combinations according to game theory”
(https://quantimschmitz.com/2023/08/12/the-best-pokemon-type-combinations-according-to-game-theory/)
with two key changes:

1.  Non-STAB coverage: A Pokémon is assumed to have at least one neutral
    attacking option (1x) even when none of its own types are useful.

2.  Defensive weighting:  Incoming damage is multiplied by DEF_WT (≥1) before
    the payoff is computed.  DEF_WT > 1 rewards combinations whose resistances
    matter more than their super-effective coverage.

Author:  Kian (<your Git username>) - 2025-05-09
"""

import itertools
import numpy as np

# ---------------------------------------------------------------------
# 1. The modern 18-type chart
#    Source: Pokémon DB type chart  (https://pokemondb.net/type)  :contentReference[oaicite:0]{index=0}
# ---------------------------------------------------------------------

TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"
]

# Everything that is not listed is 1x
IMMUNE   = {("Normal", "Ghost"), ("Fighting", "Ghost"), ("Psychic", "Dark"),
            ("Poison", "Steel"), ("Electric", "Ground"), ("Ground", "Flying"),
            ("Dragon", "Fairy")}
NOT_VERY = { # ½x
    ("Normal", "Rock"), ("Normal", "Steel"),
    ("Fire", "Fire"), ("Fire", "Water"), ("Fire", "Rock"), ("Fire", "Dragon"),
    ("Water", "Water"), ("Water", "Grass"), ("Water", "Dragon"),
    ("Electric", "Electric"), ("Electric", "Grass"), ("Electric", "Dragon"),
    ("Grass", "Fire"), ("Grass", "Grass"), ("Grass", "Poison"),
    ("Grass", "Flying"), ("Grass", "Bug"), ("Grass", "Dragon"), ("Grass", "Steel"),
    ("Ice", "Fire"), ("Ice", "Water"), ("Ice", "Ice"), ("Ice", "Steel"),
    ("Fighting", "Poison"), ("Fighting", "Flying"), ("Fighting", "Psychic"),
    ("Fighting", "Bug"), ("Fighting", "Fairy"),
    ("Poison", "Poison"), ("Poison", "Ground"), ("Poison", "Rock"),
    ("Poison", "Ghost"), ("Poison", "Steel"),
    ("Ground", "Bug"), ("Ground", "Grass"),
    ("Flying", "Electric"), ("Flying", "Rock"), ("Flying", "Steel"),
    ("Psychic", "Steel"),
    ("Bug", "Fire"), ("Bug", "Fighting"), ("Bug", "Poison"), ("Bug", "Flying"),
    ("Bug", "Ghost"), ("Bug", "Steel"), ("Bug", "Fairy"),
    ("Rock", "Fighting"), ("Rock", "Ground"), ("Rock", "Steel"),
    ("Ghost", "Dark"),
    ("Dragon", "Steel"),
    ("Dark", "Fighting"), ("Dark", "Dark"), ("Dark", "Fairy"),
    ("Steel", "Fire"), ("Steel", "Water"), ("Steel", "Electric"), ("Steel", "Steel"),
    ("Fairy", "Fire"), ("Fairy", "Poison"), ("Fairy", "Steel"),
}
SUPER    = { # 2x
    ("Fire", "Grass"), ("Fire", "Ice"), ("Fire", "Bug"), ("Fire", "Steel"),
    ("Water", "Fire"), ("Water", "Ground"), ("Water", "Rock"),
    ("Electric", "Water"), ("Electric", "Flying"),
    ("Grass", "Water"), ("Grass", "Ground"), ("Grass", "Rock"),
    ("Ice", "Grass"), ("Ice", "Ground"), ("Ice", "Flying"), ("Ice", "Dragon"),
    ("Fighting", "Normal"), ("Fighting", "Ice"), ("Fighting", "Rock"),
    ("Fighting", "Dark"), ("Fighting", "Steel"),
    ("Poison", "Grass"), ("Poison", "Fairy"),
    ("Ground", "Fire"), ("Ground", "Electric"), ("Ground", "Poison"),
    ("Ground", "Rock"), ("Ground", "Steel"),
    ("Flying", "Fighting"), ("Flying", "Bug"), ("Flying", "Grass"),
    ("Psychic", "Fighting"), ("Psychic", "Poison"),
    ("Bug", "Grass"), ("Bug", "Psychic"), ("Bug", "Dark"),
    ("Rock", "Fire"), ("Rock", "Ice"), ("Rock", "Flying"), ("Rock", "Bug"),
    ("Ghost", "Ghost"), ("Ghost", "Psychic"),
    ("Dragon", "Dragon"),
    ("Dark", "Ghost"), ("Dark", "Psychic"),
    ("Steel", "Ice"), ("Steel", "Rock"), ("Steel", "Fairy"),
    ("Fairy", "Fighting"), ("Fairy", "Dragon"), ("Fairy", "Dark"),
}

def mult(attack, defend):
    """Return the type effectiveness multiplier (float) for one-on-one."""
    if (attack, defend) in IMMUNE:
        return 0.0
    if (attack, defend) in NOT_VERY:
        return 0.5
    if (attack, defend) in SUPER:
        return 2.0
    return 1.0

# ---------------------------------------------------------------------
# 2. Enumerate all 162 modern type combinations
#    Order is alphabetical purely for reproducibility
# ---------------------------------------------------------------------

COMBOS = list(itertools.combinations_with_replacement(TYPES, 2))

INDEX = {c: i for i, c in enumerate(COMBOS)}

# ---------------------------------------------------------------------
# 3. Model parameters
#    DEF_WT > 1 ⇒ defense matters more (incoming hits “hurt” more)
# ---------------------------------------------------------------------

DEF_WT       = 1.0         # try 1.0 (original), 1.5, 2.0 …

# ---------------------------------------------------------------------
# 4. Helper: best attack a combo can throw at a defender
#    Non-STAB coverage rule: floor at 1x
# ---------------------------------------------------------------------

def best_attack(attacker, defender):
    a1, a2 = attacker
    d1, d2 = defender

    def eff(atk):
        """STAB-boosted effectiveness of one attacking type."""
        mult1 = mult(atk, d1)
        # If the defender’s two slots are the *same* (Water/Water, Ghost/Ghost, …)
        # don’t multiply the chart bonus twice – it would inflate to 4×.
        mult2 = mult(atk, d2) if d2 != d1 else 1.0
        return 1.5 * mult1 * mult2

    stab1 = eff(a1)

    # Avoid doing the same calculation twice for monotypes like (Water, Water)
    stab2 = eff(a2) if a2 != a1 else 0.0

    coverage = 1.5          # always available neutral move

    return max(stab1, stab2, coverage)

# ---------------------------------------------------------------------
# 5. Build payoff matrix with defensive weighting
#    Score(i, j) ∈ [-1, 1]  (antisymmetric)
# ---------------------------------------------------------------------

M = np.zeros((len(COMBOS), len(COMBOS)), dtype=float)

for i, c_i in enumerate(COMBOS):
    for j, c_j in enumerate(COMBOS):
        if i == j:
            continue
        atk_i = best_attack(c_i, c_j)
        atk_j = best_attack(c_j, c_i)

        # incoming damage is up-weighted
        adj_i = atk_i / (atk_j ** DEF_WT)
        adj_j = atk_j / (atk_i ** DEF_WT)

        # “winner keeps leftover HP” scoring (scaled to ±1)
        if adj_i > adj_j:
            score = 1 - (adj_j / adj_i)
        elif adj_j > adj_i:
            score = - (1 - (adj_i / adj_j))
        else:
            score = 0.0
        M[i, j] = score
        M[j, i] = -score  # antisymmetric by construction

# ---------- 6. Compute a mixed-strategy Nash equilibrium (fast LP) ----------
from scipy.optimize import linprog

N = len(COMBOS)

# Variables: p_0 … p_{N-1}  and  v  (the game value)
# We'll maximise v, so minimise -v.
c = np.zeros(N + 1)
c[-1] = -1                         # minimise -v  ⇔  maximise v

# Inequalities:   M^T p  ≥  v·1   ⇒   -M^T p + v·1 ≤ 0
A_ub = np.hstack([-M.T, np.ones((N, 1))])
b_ub = np.zeros(N)

# Equalities: probabilities sum to 1
A_eq = np.hstack([np.ones((1, N)), [[0]]])
b_eq = [1]

bounds = [(0, None)] * N + [(None, None)]   # p_i ≥ 0,  v free

res = linprog(c, A_ub, b_ub, A_eq, b_eq, method="highs", options={"disp": False})
if not res.success:
    raise RuntimeError(res.message)

sigma = res.x[:-1]               # the equilibrium distribution
game_value = res.x[-1]

# ---------------------------------------------------------------------
# 7. Rank the combinations
# ---------------------------------------------------------------------

ranking = sorted(
    ((COMBOS[i], p) for i, p in enumerate(sigma) if p > 0),
    key=lambda x: -x[1]
)

print(f"Top {len(ranking)} type combinations (DEF_WT={DEF_WT}):")
for (t1, t2), p in ranking[:27]:        # 27 for easy comparison with Schmitz
    print(f"{t1}/{t2:<7}  -  {p:.4f}")
