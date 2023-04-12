import openmm as mm
import numpy as np
from openmm import app
from openmm import unit
from openmmforcefields.generators import SystemGenerator
from openff.toolkit.topology import Molecule, Topology

from rdkit import Chem
from rdkit.Chem import AllChem

lig_file = "/home/boris/Data/BigBindStructV2/AL7A1_HUMAN_27_537_0/4zul_un1_lig.sdf"
rec_file = "out.pqr" # "/home/boris/Data/BigBindStructV2/AL7A1_HUMAN_27_537_0/4zul_A_rec.pdb"
poc_file = "/home/boris/Data/BigBindStructV2/AL7A1_HUMAN_27_537_0/4zul_A_rec_pocket.pdb"

# freeze all atoms not in the pocket
res_ids = set()
with open(poc_file, "r") as f:
    for line in f.readlines():
        if line.startswith("ATOM"):
            res_ids.add(int(line[22:26]))

# Load the protein from a PDB file
pdb = app.PDBFile(rec_file)
lig = Molecule.from_file(lig_file)
lig.compute_partial_charges()

lig_coord = lig.conformers[0] + 2*np.random.randn(*lig.conformers[0].shape)*unit.angstrom

modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.add(lig.to_topology().to_openmm(), lig_coord)
mergedTopology = modeller.topology
mergedPositions = modeller.positions

forcefield_kwargs = {
    "nonbondedCutoff": 1*unit.nanometer,
    "constraints": "HBonds"
}
ffs = ['amber/ff14SB.xml', 'amber/tip3p_standard.xml']
 #ffs = [ "charmm36.xml" ]
system_generator = SystemGenerator(forcefields=ffs, small_molecule_forcefield='gaff-2.11', molecules=[lig], forcefield_kwargs=forcefield_kwargs, cache='db.json')
# Create an OpenMM System from an Open Force Field toolkit Topology object
system = system_generator.create_system(mergedTopology)

for r in modeller.topology.residues():
    if int(r.id) in res_ids or r.chain.index == 1:
        print(r)
        continue
    for a in r.atoms():
        system.setParticleMass(a.index, 0.0)


# Set up the integrator for energy minimization
integrator = mm.VerletIntegrator(0.001*unit.picoseconds)

# Set up the simulation context with the system, integrator, and positions from the PDB file
context = mm.Context(system, integrator)
context.setPositions(mergedPositions)

# Minimize the energy of the system
print('Minimizing energy...')
mm.LocalEnergyMinimizer.minimize(context)

state = context.getState(getEnergy=True, getForces=True)
energy = state.getPotentialEnergy()
print(energy)

# Get the minimized positions and save them to a PDB file
positions = context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(mergedTopology, positions, open('minimized.pdb', 'w'))