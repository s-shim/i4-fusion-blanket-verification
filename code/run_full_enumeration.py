import itertools
import json
import math
import time
import numpy as np
import openmc

# ================= Fixed geometry (reused from i4_case_v3.py) =================
DAGMC_FILE = "i4_blanket_v3.h5m"
R = 300.0
r_out = 20.0
sector_names = [f"sector{i}" for i in range(4)]

# ================= K=4 candidate recipes (Li-6 enrichment) =================
recipes = [0.15, 0.30, 0.60, 0.90]
K = len(recipes)
I = 4  # number of sectors

# ================= Closed-form Cost(x): no oracle needed =================
V_sector_cm3 = (math.pi ** 2 * R * r_out ** 2) / 2   # 90-degree torus sector volume
V_sector_m3 = V_sector_cm3 / 1e6
mass_sector_kg = V_sector_m3 * 10000.0  # PbLi density = 10,000 kg/m^3

c_base = 12.95     # $/kg, natural PbLi base price
f_Li = 0.00681     # Li mass fraction in PbLi
c_curve = {0.15: 1150, 0.30: 2300, 0.60: 3150, 0.90: 3550}  # $/kg of enriched Li-6

def sector_cost(eps):
    c_blend = c_base + f_Li * eps * c_curve[eps]
    return mass_sector_kg * c_blend

recipe_cost = {eps: sector_cost(eps) for eps in recipes}
print(">>> Per-sector cost by recipe:", {f"{e:.2f}": f"${c:,.0f}" for e, c in recipe_cost.items()})

def total_cost(assignment):
    return sum(recipe_cost[eps] for eps in assignment)

# ================= Fixed settings/tallies (written once, reused every run) =================
r_dist = openmc.stats.Discrete([R], [1.0])
phi_dist = openmc.stats.Uniform(0, 2 * np.pi)
z_dist = openmc.stats.Discrete([0.0], [1.0])
space = openmc.stats.CylindricalIndependent(r=r_dist, phi=phi_dist, z=z_dist, origin=(0., 0., 0.))
energy = openmc.stats.Discrete([14.1e6], [1.0])
source = openmc.IndependentSource(space=space, angle=openmc.stats.Isotropic(), energy=energy)

settings = openmc.Settings()
settings.source = source
settings.batches = 20
settings.particles = 5000
settings.run_mode = 'fixed source'
settings.export_to_xml()

dag_univ = openmc.DAGMCUniverse(DAGMC_FILE).bounded_universe()
geometry = openmc.Geometry(dag_univ)
geometry.export_to_xml()

tbr_tally = openmc.Tally(name='TBR')
tbr_tally.scores = ['H3-production']
tallies = openmc.Tallies([tbr_tally])
tallies.export_to_xml()

# ================= Enumerate all 4^4 = 256 configurations =================
all_configs = list(itertools.product(range(K), repeat=I))
print(f">>> Total configurations: {len(all_configs)}")

results = []
t_start = time.time()

for call_idx, combo in enumerate(all_configs):
    assignment = [recipes[k] for k in combo]  # e.g. [0.15, 0.30, 0.15, 0.90]

    # ---- Materials for this configuration ----
    materials_list = []
    for i, eps in enumerate(assignment):
        m = openmc.Material(name=sector_names[i])
        m.add_element('Pb', 84.3, percent_type='ao')
        m.add_element('Li', 15.7, percent_type='ao',
                      enrichment=eps * 100, enrichment_target='Li6', enrichment_type='ao')
        m.set_density('g/cm3', 10.0)
        materials_list.append(m)
    openmc.Materials(materials_list).export_to_xml()

    # ---- Cost(x): closed-form, free ----
    cost = total_cost(assignment)

    # ---- TBR(x): real OpenMC oracle call ----
    openmc.run(output=False)
    sp = openmc.StatePoint('statepoint.20.h5')
    tbr_tally_result = sp.get_tally(name='TBR')
    tbr_mean = float(tbr_tally_result.mean.flatten()[0])
    tbr_std = float(tbr_tally_result.std_dev.flatten()[0])
    sp.close()

    record = {
        "call": call_idx,
        "combo_indices": list(combo),
        "recipe_assignment": assignment,
        "TBR": tbr_mean,
        "TBR_std": tbr_std,
        "Cost": cost,
    }
    results.append(record)

    with open(f"results/call_{call_idx:04d}.json", "w") as f:
        json.dump(record, f, indent=2)

    elapsed = time.time() - t_start
    print(f"[{call_idx+1}/{len(all_configs)}] assignment={assignment} "
          f"TBR={tbr_mean:.4f}+/-{tbr_std:.4f} Cost=${cost:,.0f} "
          f"(elapsed {elapsed:.0f}s)")

# ================= Compute true Pareto frontier (maximize TBR, minimize Cost) =================
def is_dominated(a, b):
    # b dominates a if b is at least as good in both objectives and strictly better in one
    return (b["TBR"] >= a["TBR"] and b["Cost"] <= a["Cost"]
            and (b["TBR"] > a["TBR"] or b["Cost"] < a["Cost"]))

pareto = []
for a in results:
    if not any(is_dominated(a, b) for b in results if b is not a):
        pareto.append(a)

pareto_sorted = sorted(pareto, key=lambda r: r["Cost"])

summary = {
    "n_configs": len(results),
    "all_results": results,
    "pareto_frontier": pareto_sorted,
}
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n>>> Done. {len(all_configs)} configurations evaluated in {time.time()-t_start:.0f}s.")
print(f">>> True Pareto frontier has {len(pareto_sorted)} points:")
for p in pareto_sorted:
    print(f"    Cost=${p['Cost']:,.0f}  TBR={p['TBR']:.4f}+/-{p['TBR_std']:.4f}  recipe={p['recipe_assignment']}")
