import csv
import itertools
import math
import numpy as np
from typing import Dict, Tuple, List
from numpy.typing import NDArray

from pokemon import Pokemon


# Pokemon Type Chart
def csv_to_dict(filepath: str) -> Dict[str, Dict[str, float]]:
    matchups: Dict[str, Dict[str, float]] = {}
    with open(filepath, newline="") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)[1:]
        for row in reader:
            row_name = row[0]
            matchups[row_name] = {col: float(val) for col, val in zip(headers, row[1:])}
    return matchups


matchups = csv_to_dict("./data/matchups.csv")
moves = csv_to_dict("./data/moves.csv")

# All pokemon type combinations
pokemon_roster: List[Pokemon] = [Pokemon((t,)) for t in matchups.keys()] + [
    Pokemon(combo) for combo in itertools.combinations(matchups.keys(), 2)
]


# Helper function for determining damage multiplier and payoff
def damage_dealt(attacker: Pokemon, defender: Pokemon) -> float:
    best = 60.0  # weak non-stab coverage move

    for move_type in attacker.types:
        mv = moves[move_type]
        power = mv["power"] * mv["accuracy"]
        stab = 1.5
        eff = math.prod(matchups[move_type][d] for d in defender.types)
        dmg = power * stab * eff * (1 - mv["recoil"])
        best = max(best, dmg)

    return best


def payoff(attacker: Pokemon, defender: Pokemon) -> float:
    avd = damage_dealt(attacker, defender)
    dva = damage_dealt(defender, attacker)
    return (1 + (avd - dva) / max(avd, dva)) / 2


# Create the Payoff matrix
payoff_matrix: List[List[float]] = [
    [payoff(a, d) for d in pokemon_roster] for a in pokemon_roster
]


# Estimate the nash equilibirum mixed strategy
def nash_equilibrium_estimator(
    payoff_matrix: List[List[float]],
    roster: List[Pokemon],
    std_threshold: float = 0.001,
    max_iter: int = 30_000,
) -> List[Tuple[Pokemon, int, float]]:
    n = len(roster)
    pop_array = np.ones(n, dtype=np.int64)
    payoff_array = np.array(payoff_matrix, dtype=np.float64)
    estimated_wr = np.zeros(n, dtype=np.float64)

    std_dev = float("inf")
    iter_count = 0

    while std_dev > std_threshold and iter_count < max_iter:
        iter_count += 1

        estimated_wr: NDArray[np.float64] = (payoff_array @ pop_array) / pop_array.sum()

        # Only look at strategies that are being played
        played_wr = estimated_wr[pop_array > 0]

        mean: np.float64 = played_wr.mean()
        std_dev = float(played_wr.std(ddof=0)) if len(played_wr) > 1 else 0.0

        # Vectorized population update
        pop_array += (estimated_wr > mean).astype(int)
        pop_array -= ((estimated_wr < (mean - std_dev)) & (pop_array > 0)).astype(int)

    result = [
        (roster[i], int(pop_array[i]), float(estimated_wr[i]))
        for i in range(n)
        if pop_array[i] > 0
    ]

    result.sort(key=lambda x: x[1], reverse=True)
    return result


if __name__ == "__main__":
    equilibrium = nash_equilibrium_estimator(payoff_matrix, pokemon_roster)

    total_pop = sum(pop for _, pop, _ in equilibrium)
    for poke, pop, wr in equilibrium:
        pct = pop / total_pop * 100
        print(f"{'/'.join(poke.types):12}: pop={pct:5.2f}%, est_wr={wr:.3f}")
