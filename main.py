"""
Re-implementation of
“The best Pokémon type combinations according to game theory”
(https://quantimschmitz.com/2023/08/12/the-best-pokemon-type-combinations-according-to-game-theory/)
with two key changes:

1.  Non-STAB coverage: A Pokémon is assumed to have at least one neutral
    attacking option (1×) even when none of its own types are useful.

2.  Defensive weighting:  Incoming damage is multiplied by DEF_WT (≥1) before
    the payoff is computed.  DEF_WT > 1 rewards combinations whose resistances
    matter more than their super-effective coverage.

Author:  Kian (<your Git username>) – 2025-05-09
"""

import itertools
import numpy as np
import nashpy as nash

# ---------------------------------------------------------------------
# 1. The modern 18-type chart
#    Source: Pokémon DB type chart  (https://pokemondb.net/type)  :contentReference[oaicite:0]{index=0}
# ---------------------------------------------------------------------

TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"
]

# Everything that is not listed is 1×
IMMUNE   = {("Normal", "Ghost"), ("Fighting", "Ghost"), ("Psychic", "Dark"),
            ("Poison", "Steel"), ("Electric", "Ground"), ("Ground", "Flying"),
            ("Dragon", "Fairy")}
NOT_VERY = { # ½×
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
SUPER    = { # 2×
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
    """Return the type effectiveness multiplier (float) for one‐on‐one."""
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

COMBOS = []
for a, b in itertools.combinations_with_replacement(TYPES, 2):
    COMBOS.append((a, b))

INDEX = {c: i for i, c in enumerate(COMBOS)}

# ---------------------------------------------------------------------
# 3. Model parameters
#    DEF_WT > 1 ⇒ defense matters more (incoming hits “hurt” more)
# ---------------------------------------------------------------------

DEF_WT       = 1.5          # try 1.0 (original), 1.5, 2.0 …
N_ITERATIONS = 30           # number of random starts for equilibrium search
RNG_SEED     = 42

# ---------------------------------------------------------------------
# 4. Helper: best attack a combo can throw at a defender
#    Non-STAB coverage rule: floor at 1×
# ---------------------------------------------------------------------

def best_attack(attacker, defender):
    a1, a2 = attacker
    d1, d2 = defender
    effs = [
        mult(a1, d1) * mult(a1, d2),
        mult(a2, d1) * mult(a2, d2) if a2 != a1 else 0,
    ]
    return max(max(effs), 1.0)   # ≥ 1× because of coverage

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

# ---------------------------------------------------------------------
# 6. Compute an empirical mixed-strategy Nash equilibrium
#    We do multiple random restarts because the matrix is huge (324×324)
# ---------------------------------------------------------------------

np.random.seed(RNG_SEED)
agg_probs = np.zeros(len(COMBOS), dtype=float)

for _ in range(N_ITERATIONS):
    A = nash.Game(M)
    sigma_i, sigma_j = A.support_enumeration().__next__()
    # Any two-player zero-sum equilibrium gives identical row/col strategies
    agg_probs += sigma_i

avg_probs = agg_probs / N_ITERATIONS

# ---------------------------------------------------------------------
# 7. Rank the combinations
# ---------------------------------------------------------------------

ranking = sorted(
    ((COMBOS[i], p) for i, p in enumerate(avg_probs) if p > 0),
    key=lambda x: -x[1]
)

print(f"Top {len(ranking)} type combinations (DEF_WT={DEF_WT}):")
for (t1, t2), p in ranking[:27]:        # 27 for easy comparison with Schmitz
    print(f"{t1}/{t2:<7}  –  {p:.4f}")
