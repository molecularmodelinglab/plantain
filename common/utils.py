import os
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

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = np.eye(4)
    M[:-1,:-1] = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

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
                print(f"Error parsing cached {cache_fname}, contining loading file")
                traceback.print_exc()

        structure = PDBParser().get_structure("random_id", fname)
        io=MMTFIO()
        io.set_structure(structure)
        io.save(cache_fname)

        ret = structure[0]
        return ret

def get_mol_from_file_no_cache(fname):
    """ Loads sdf and pdbqt files, returns rdkit mol"""
    if fname.endswith(".pdbqt"):
        mol = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule.from_file(fname))[0]
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
                return ret
        except KeyboardInterrupt:
            raise
        except:
            print(f"Error parsing cached {cache_fname}, contining loading file")
            traceback.print_exc()

    ret = get_mol_from_file_no_cache(fname)
    with open(cache_fname, "wb") as f:
        pickle.dump(ret, f)

    return ret

            