import os
import re
import traceback
import warnings
import numpy as np
import pickle
from meeko import RDKitMolCreate, PDBQTMolecule
from Bio.PDB.mmtf import MMTFIO, MMTFParser
from rdkit import Chem
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser

def get_activity(batch):
    """ For ease, we also want to support data coming in X, Y tuples """
    if isinstance(batch, tuple):
        return batch[1]
    else:
        return batch.activity

def get_prot_from_file_no_cache(fname):
    """ Loads pdb files, return first biopython chain"""
    assert fname.endswith(".pdb")
    parser = PDBParser()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = parser.get_structure('random_id', fname)
        rec = structure[0]
        
    return rec

# Somewhat hackily added automatic caching for both molecule and protein file formats
# proteins cache to mmtf files, molecules ot pickles

def get_prot_from_file(fname, cache=True):
    if not cache:
        return get_mol_from_file_no_cache(fname)
    cache_fname = ".".join(fname.split(".")[:-1]) + ".mmtf"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        
        if os.path.exists(cache_fname):
            try:
                with open(cache_fname, "rb") as fh:
                    structure = MMTFParser().get_structure(cache_fname)
                    ret = structure[0]
                    return ret
            except KeyboardInterrupt:
                raise
            except:
                print(f"Error parsing cached {cache_fname}, continuing loading file")
                traceback.print_exc()

        structure = PDBParser().get_structure("random_id", fname)
        io=MMTFIO()
        io.set_structure(structure)
        io.save(cache_fname)

        ret = structure[0]
        return ret

smiles_re = re.compile(".*REMARK SMILES\s+(.+)")
def get_mol_from_file_no_cache(fname):
    """ Loads sdf and pdbqt files, returns rdkit mol"""
    if fname.endswith(".pdbqt"):
        pdbqt = PDBQTMolecule.from_file(fname)
        # manually fix to account for bug in gnina not writing newlines
        # after REMARKS
        if pdbqt._pose_data["smiles"][0] is None:
            with open(fname, "r") as f:
                for line in f.readlines():
                    m = smiles_re.match(line)
                    if m is not None:
                        pdbqt._pose_data["smiles"][0] = m.groups()[0]
                        break
        mol = RDKitMolCreate.from_pdbqt_mol(pdbqt)[0]
    elif fname.endswith(".sdf"):
        mol = next(Chem.SDMolSupplier(fname, sanitize=True))
    else:
        raise ValueError(f"invalid file extension for {fname}")
    return mol

def get_mol_from_file(fname, cache=True):
    if not cache:
        return get_mol_from_file_no_cache(fname)
    cache_fname = ".".join(fname.split(".")[:-1]) + ".pkl"

    if os.path.exists(cache_fname):
        try:
            with open(cache_fname, "rb") as fh:
                ret = pickle.load(fh)
                if ret is None:
                    raise Exception("Molecule is None")
                return ret
        except KeyboardInterrupt:
            raise
        except:
            pass
            # print(f"Error parsing cached {cache_fname}, continuing loading file")
            # traceback.print_exc()

    ret = get_mol_from_file_no_cache(fname)
    with open(cache_fname, "wb") as f:
        pickle.dump(ret, f)

    return ret

def get_docked_scores_from_pdbqt(fname):
    return PDBQTMolecule.from_file(fname)._pose_data["free_energies"]


score_re = re.compile(".*CNNscore\s+([\d|\.]+)")
affinity_re = re.compile(".*CNNaffinity\s+([\d|\.]+)")
def get_gnina_scores_from_pdbqt(fname):
    """ returns pose scores and affinities for all the poses in
    pdbqt file fname """
    scores = []
    affinities = []
    with open(fname) as f:
        for line in f.readlines():
            m = score_re.match(line)
            if m is not None:
                scores.append(float(m.groups()[0]))
            m = affinity_re.match(line)
            if m is not None:
                affinities.append(float(m.groups()[0]))
    return scores, affinities

def flatten_dict(d):
    ret = {}
    for key, val in d.items():
        if isinstance(val, dict):
            for key2, val2 in flatten_dict(val).items():
                ret[f"{key}_{key2}"] = val2
        else:
            ret[key] = val
    return ret

            