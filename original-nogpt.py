import csv
import itertools
from typing import Dict, Tuple, List
import numpy as np

# Pokemon Type Chart
def load_matchups(filepath: str) -> Dict[str, Dict[str, float]]:
    matchups: Dict[str, Dict[str, float]] = {}
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)[1:]
        for row in reader:
            row_name = row[0]
            matchups[row_name] = {col: float(val) for col, val in zip(headers, row[1:])}
    return matchups

matchups = load_matchups("matchups.csv")

# All pokemon types
types = (
    [[t] for t in matchups.keys()]
    + [list(combo) for combo in itertools.combinations(matchups.keys(), 2)]
)

# Helper function for determining damage multiplier and payoff
def damage_dealt(attacker: List[str], defender: List[str]) -> float:
    best = 0.0
    for atk in attacker:
        eff = 1.0
        for d in defender:
            eff *= matchups[atk][d]
        best = max(best, eff)
    return best

def payoff(attacker: List[str], defender: List[str]) -> float:
    avd = damage_dealt(attacker, defender)
    dva = damage_dealt(defender, attacker)
    if avd == 0 and dva == 0: return 0.5
    return (1 + (avd - dva) / max(avd, dva)) / 2
    
# Create the Payoff matrix
payoff_matrix = [
    [payoff(attacker, defender) for defender in types]
    for attacker in types
]

def get_payoff(
    attacker: List[str],
    defender: List[str],
    types_list: List[List[str]] = types,
    matrix: List[List[float]] = payoff_matrix
) -> float:
    i = types_list.index(attacker)
    j = types_list.index(defender)
    return matrix[i][j]

# Estimate the nash equilibirum mixed strategy
def nash_equilibrium_estimator(
    payoff_matrix: List[List[float]],
    types: List[List[str]],
    std_threshold: float = 0.001
) -> List[Tuple[List[str], int, float]]:
    n = len(types)
    pop: np.ndarray = np.ones(n, dtype=int)

    std_dev = float('inf')
    estimated_wr = np.zeros(n, dtype=float)

    while std_dev > std_threshold:
        total_pop = pop.sum()
        estimated_wr = np.dot(payoff_matrix, pop) / total_pop

        played = pop > 0
        played_wr = estimated_wr[played]

        mean = played_wr.mean()
        std_dev = played_wr.std()

        for i in range(n):
            if estimated_wr[i] > mean:
                pop[i] += 1
            elif estimated_wr[i] < mean - std_dev and pop[i] > 0:
                pop[i] -= 1

    result: List[Tuple[List[str], int, float]] = []
    for i in range(n):
        if pop[i] > 0:
            result.append((types[i], int(pop[i]), float(estimated_wr[i])))

    result.sort(key=lambda x: x[1], reverse=True)
    return result

# Calculate Nash Equilibrium
equilibrium = nash_equilibrium_estimator(payoff_matrix, types)

# Print the results
total_pop = sum(popularity for _, popularity, _ in equilibrium)
for combo, popularity, ewr in equilibrium:
    pct = popularity / total_pop * 100
    types_str = "/".join(combo)
    print(f"{types_str:12s}: pop={pct:5.2f}%, est_wr={ewr:.3f}")