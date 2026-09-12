import numpy as np
import cadquery as cq
import paramak
from cad_to_dagmc import CadToDagmc
import openmc

# ================= Geometry parameters =================
R = 300.0      # major radius (cm)
r_out = 20.0   # tube cross-section radius (cm) -- solid, no inner hole
N = 16         # points approximating the circular cross-section

# Circular cross-section approximated by N straight segments, centered at (R, 0)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
profile = [(R + r_out * np.cos(t), r_out * np.sin(t), "straight") for t in theta]

sector_names = [f"sector{i}" for i in range(4)]
assembly = cq.Assembly()
for i in range(4):
    solid = paramak.revolved_shape(
        points=profile,
        rotation_angle=90.0,
        plane="XZ",
        name=sector_names[i],
    )
    rotated = solid.rotate((0, 0, 0), (0, 0, 1), 90.0 * i)
    assembly.add(rotated, name=sector_names[i])

# ================= Convert to DAGMC =================
model = CadToDagmc()
model.add_cadquery_object(cadquery_object=assembly, material_tags="assembly_names")
model.export_dagmc_h5m_file(filename="i4_blanket_v3.h5m")
print(">>> DAGMC file written: i4_blanket_v3.h5m")

# ================= Materials =================
test_assignment = [0.15, 0.30, 0.60, 0.90]
materials_list = []
for i, eps in enumerate(test_assignment):
    m = openmc.Material(name=sector_names[i])
    m.add_element('Pb', 84.3, percent_type='ao')
    m.add_element('Li', 15.7, percent_type='ao',
                  enrichment=eps * 100, enrichment_target='Li6', enrichment_type='ao')
    m.set_density('g/cm3', 10.0)
    materials_list.append(m)
materials = openmc.Materials(materials_list)
materials.export_to_xml()

# ================= Geometry from DAGMC =================
dag_univ = openmc.DAGMCUniverse("i4_blanket_v3.h5m").bounded_universe()
geometry = openmc.Geometry(dag_univ)
geometry.export_to_xml()

# ================= Source: ring at major radius R =================
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

# ================= Tally =================
tbr_tally = openmc.Tally(name='TBR')
tbr_tally.scores = ['H3-production']
tallies = openmc.Tallies([tbr_tally])
tallies.export_to_xml()

# ================= Run =================
openmc.run()

sp = openmc.StatePoint('statepoint.20.h5')
tbr = sp.get_tally(name='TBR')
print(f"\n>>> TBR = {tbr.mean.flatten()[0]:.4f} +/- {tbr.std_dev.flatten()[0]:.4f}")
print(f">>> Recipe assignment: {dict(zip(sector_names, test_assignment))}")
