import os
import subprocess
import sys
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from common.cfg_utils import get_config
from common.pose_transform import Pose, PoseTransform, add_pose_to_mol
from common.torsion import TorsionData
from common.utils import get_mol_from_file
from data_formats.graphs.mol_graph import get_mol_coords
import openmm as mm
import numpy as np
from openmm import app
from openmm import unit
from openmmforcefields.generators import SystemGenerator
from openff.toolkit.topology import Molecule, Topology
from terrace.batch import collate
from terrace.dataframe import DFRow

def get_lig_and_poses(cfg, lig_file):
    lig = get_mol_from_file(lig_file)
    lig = Chem.AddHs(lig, addCoords=True)
    lig_crystal_pose = Pose(get_mol_coords(lig, 0))
    lig_tor_data = TorsionData.from_mol(lig)
    x = collate([DFRow(lig=lig, lig_crystal_pose=lig_crystal_pose, lig_torsion_data=lig_tor_data)])
    diff_transforms = PoseTransform.make_diffused(cfg.model.diffusion, cfg.model.diffusion.timesteps, x, "cpu")
    diff_poses = diff_transforms[0].apply(lig_crystal_pose, lig_tor_data)
    return lig, diff_poses

def setup_openmm(lig, pqr_file, poc_file):

    # find all the pocket residues so we can freeze all the non-pocket residues
    poc_res_ids = set()
    with open(poc_file, "r") as f:
        for line in f.readlines():
            if line.startswith("ATOM"):
                poc_res_ids.add(int(line[22:26]))

    # Load the protein from a PDB file
    rec = app.PDBFile(pqr_file)
    oflig = Molecule.from_rdkit(lig)

    modeller = app.Modeller(rec.topology, rec.positions)
    modeller.add(oflig.to_topology().to_openmm(), oflig.conformers[0])
    mergedTopology = modeller.topology
    mergedPositions = modeller.positions

    forcefield_kwargs = {
        "nonbondedCutoff": 1*unit.nanometer,
        "constraints": "HBonds"
    }
    ffs = ['amber/ff14SB.xml', 'amber/tip3p_standard.xml']
    system_generator = SystemGenerator(forcefields=ffs, small_molecule_forcefield='gaff-2.11', molecules=[oflig], forcefield_kwargs=forcefield_kwargs, cache='db.json')
    system = system_generator.create_system(mergedTopology)

    for r in modeller.topology.residues():
        if int(r.id) in poc_res_ids or r.chain.index == 1:
            continue
        for a in r.atoms():
            system.setParticleMass(a.index, 0.0)

    # Set up the integrator for energy minimization
    integrator = mm.VerletIntegrator(0.001*unit.picoseconds)

    # Set up the simulation context with the system, integrator, and positions from the PDB file
    context = mm.Context(system, integrator)

    return context, mergedTopology, mergedPositions

def process_row(cfg, row):
    lig_file = cfg.platform.bigbind_struct_v2_dir + "/" + row.lig_crystal_file
    rec_file = cfg.platform.bigbind_struct_v2_dir + "/" + row.redock_rec_file
    poc_file = cfg.platform.bigbind_struct_v2_dir + "/" + row.redock_rec_pocket_file
    pqr_file = cfg.platform.bigbind_struct_ff_dir + "/" + row.redock_rec_file.split(".")[0] + ".pqr" 
    
    diff_file_prefix = ".".join(pqr_file.split(".")[:-1]) + "_diff_"
    lig_diff_prefix = cfg.platform.bigbind_struct_ff_dir + "/" + ".".join(row.lig_crystal_file.split(".")[:-1]) + "_diff_"

    # use pdb2pqr to clean up the pdb file, if we haven't already done that
    if not os.path.exists(pqr_file):
        poc_folder = "/".join(pqr_file.split("/")[:-1])
        os.makedirs(poc_folder, exist_ok=True)
        cmd = [ "pdb2pqr30", rec_file, pqr_file ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        out, err = proc.communicate()

    # get the diffused poses
    n_poses = cfg.model.diffusion.timesteps
    lig, poses = get_lig_and_poses(cfg, lig_file)

    # set up openmm system
    context, mergedTopology, mergedPositions = setup_openmm(lig, pqr_file, poc_file)

    energies = []
    for p in range(n_poses):
        lig_coords = poses.get(p).coord.numpy()
        for i, coord in enumerate(lig_coords):
            merged_idx = len(mergedPositions) - len(lig_coords) + i
            # print(p, i, mergedPositions[merged_idx] - coord*unit.angstrom)
            mergedPositions[merged_idx] = coord*unit.angstrom
        
        context.setPositions(mergedPositions)
        mm.LocalEnergyMinimizer.minimize(context)

        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy()
        energies.append(energy.value_in_unit(unit.kilojoule_per_mole))

        diff_filename = diff_file_prefix + str(p) + ".pdb"
        lig_diff_filename = lig_diff_prefix + str(p) + ".sdf"

        # Get the minimized positions and save them to a PDB file
        positions = context.getState(getPositions=True).getPositions()
        app.PDBFile.writeFile(mergedTopology, positions, open(diff_filename, 'w'))

        add_pose_to_mol(lig, poses.get(p))
        
        writer = Chem.SDWriter(lig_diff_filename)
        writer.write(lig)
        writer.close()

    return diff_file_prefix, lig_diff_prefix, energies

def main(cfg):
    df = pd.read_csv(f"{cfg.platform.bigbind_struct_v2_dir}/structures_all.csv")
    
    num_workers = int(sys.argv[1])
    cur_worker = int(sys.argv[2])
    rows_per_worker = (len(df)/num_workers + 1)
    start_idx = int(rows_per_worker*cur_worker)
    end_idx = int(rows_per_worker*(cur_worker+1))
    df = df.loc[start_idx:end_idx]
    
    out = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        diff_file_prefix, lig_diff_prefix, energies = process_row(cfg, row)
        out_row = row.to_dict()
        out_row["diff_rec_prefix"] = diff_file_prefix
        out_row["diff_lig_prefix"] = lig_diff_prefix
        for i, energy in enumerate(energies):
            out_row[f"diff_energy_{i}"] = energy
        out.append(out_row)

    out_filename = cfg.platform.bigbind_struct_ff_dir + f"/structures_all_{cur_worker}.csv"
    out_df = pd.DataFrame(out)
    out_df.to_csv(out_filename)

if __name__ == "__main__":
    cfg = get_config("diffusion_v2")
    main(cfg)