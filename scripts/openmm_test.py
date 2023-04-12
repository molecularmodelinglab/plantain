from rdkit import Chem
molOrig = Chem.MolFromSmiles("CCCCCCCCCC")
molOrig = Chem.AddHs(molOrig)
# AllChem.EmbedMultipleConfs(molOrig)
# Chem.rdmolops.AssignAtomChiralTagsFromStructure(molOrig)
from openff.toolkit.topology import Molecule
ofmol = Molecule.from_rdkit(molOrig)
#generate_unique_atom_names(ofmol)
ofmol.generate_conformers(n_conformers=10)
pos = ofmol.conformers[0]
ofmol.name = 'molecule'

from openmm import app
from openmm import unit
forcefield_kwargs = {}
# Initialize a SystemGenerator using GAFF
from openmmforcefields.generators import SystemGenerator
system_generator = SystemGenerator(forcefields=['amber/ff14SB.xml', 'amber/tip3p_standard.xml'], small_molecule_forcefield='gaff-2.11', molecules=[ofmol], forcefield_kwargs=forcefield_kwargs, cache='db.json')
# Create an OpenMM System from an Open Force Field toolkit Topology object
system = system_generator.create_system(ofmol.to_topology().to_openmm())