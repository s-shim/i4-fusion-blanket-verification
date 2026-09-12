import json
import numpy as np

recipes = [0.15, 0.30, 0.60, 0.90]
K, I = 4, 4

with open('results/summary.json') as f:
    summary = json.load(f)

lookup = {tuple(r['combo_indices']): (r['TBR'], r['TBR_std'], r['Cost'])
          for r in summary['all_results']}
true_pareto = summary['pareto_frontier']

# Global cache of real oracle calls made so far (shared across all budget sweeps)
evaluated = {}

def real_oracle_call(combo):
    """Look up the already-computed real OpenMC result for this exact
    configuration (deterministic replay of the actual full enumeration run).
    Counts as one real oracle call only the first time a given combo is seen."""
    if combo not in evaluated:
        evaluated[combo] = lookup[combo]
    return evaluated[combo]

def cost_of(combo):
    return lookup[combo][2]  # closed-form, free, no oracle needed

class TBRSurrogate:
    """TBR ~ a + b * (sum of enrichment across sectors), refit online
    after every real oracle call -- same role as the linear k_inf/CL
    surrogates in the nuclear fuel assembly case study."""
    def __init__(self):
        self.X, self.y = [], []
        self.a, self.b = 0.05, 0.15  # rough prior before any real data

    def feature(self, combo):
        return sum(recipes[k] for k in combo)

    def predict(self, combo):
        return self.a + self.b * self.feature(combo)

    def update(self, combo, tbr):
        self.X.append(self.feature(combo))
        self.y.append(tbr)
        if len(self.X) >= 2:
            X, y = np.array(self.X), np.array(self.y)
            A = np.vstack([X, np.ones_like(X)]).T
            self.b, self.a = np.linalg.lstsq(A, y, rcond=None)[0]

_fed_to_surrogate = set()

def search_for_budget(budget, surrogate, rng, max_trials=500):
    centroid = np.full((I, K), 1.0 / K)
    best_combo, best_tbr, stuck = None, -1.0, 0

    # --- Feasibility stage: explicitly check the guaranteed-cheapest combo
    # before any randomized search. The surrogate is only ever trained on
    # this point ONCE globally (not once per budget level) to avoid biasing
    # its fit with a duplicated low-TBR training point at every budget.
    cheapest_combo = tuple([0] * I)
    if cost_of(cheapest_combo) <= budget:
        tbr0, _, _ = real_oracle_call(cheapest_combo)
        if cheapest_combo not in _fed_to_surrogate:
            surrogate.update(cheapest_combo, tbr0)
            _fed_to_surrogate.add(cheapest_combo)
        best_combo, best_tbr = cheapest_combo, tbr0

    for _ in range(max_trials):
        ptb = centroid * rng.uniform(0, 1, size=(I, K))
        combo = tuple(int(np.argmax(ptb[i])) for i in range(I))
        if cost_of(combo) > budget:
            continue
        if best_combo is not None and surrogate.predict(combo) <= best_tbr:
            continue  # surrogate says "not an improvement" -> no oracle call
        tbr, _, _ = real_oracle_call(combo)
        surrogate.update(combo, tbr)
        if tbr > best_tbr:
            best_tbr, best_combo, stuck = tbr, combo, 0
        else:
            stuck += 1
        target = np.zeros((I, K))
        for i, k in enumerate(best_combo):
            target[i, k] = 1.0
        centroid = 0.7 * centroid + 0.3 * target
        if stuck > 12:
            centroid = np.full((I, K), 1.0 / K)
            stuck = 0
    return best_combo, best_tbr

budget_levels = sorted(set(r['Cost'] for r in true_pareto))
rng = np.random.default_rng(42)
surrogate = TBRSurrogate()
approx_frontier = []

for budget in budget_levels:
    combo, tbr = search_for_budget(budget, surrogate, rng)
    approx_frontier.append({"budget": budget, "combo": combo, "TBR": tbr})

print(f">>> Real oracle calls used: {len(evaluated)} out of 256 possible configurations\n")
print(f"{'Budget ($)':>12} {'ARR-found TBR':>14} {'True optimal TBR':>18} {'Gap':>8}  Recipe")
true_by_cost = {r['Cost']: r['TBR'] for r in true_pareto}
for a in approx_frontier:
    true_tbr = true_by_cost.get(a['budget'], None)
    if a['combo'] is None:
        print(f"{a['budget']:>12,.0f} {'--- no feasible combo found within max_trials ---':>50}")
        continue
    gap = (true_tbr - a['TBR']) if true_tbr is not None else float('nan')
    recipe = [recipes[k] for k in a['combo']]
    print(f"{a['budget']:>12,.0f} {a['TBR']:>14.4f} {true_tbr:>18.4f} {gap:>8.4f}  {recipe}")
