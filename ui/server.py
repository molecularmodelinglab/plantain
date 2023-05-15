from flask import Flask, render_template, request, jsonify
from io import StringIO, BytesIO
from rdkit import Chem
import json

app = Flask(__name__, template_folder=".")

@app.route('/')
def index():
    # Render the main HTML template
    return render_template('index.html')

def sdf_to_mol(sdf_str):
    """ SDF file contents to rdkit mol """
    sio = BytesIO()
    sio.write(sdf_str.encode('utf8'))
    sio.seek(0)
    return next(Chem.ForwardSDMolSupplier(sio, sanitize=True))

def mol_to_sdf(mol):
    """ rdkit mol to SDF file contents """
    sio = StringIO()
    writer = Chem.SDWriter(sio)
    writer.write(mol)
    writer.close()

    sio.seek(0)
    return sio.read()

@app.route('/init_mol')
def init_mol():
    # Load a sample molecule from a file
    molfile = "/home/boris/Data/BigBindStructV2/AL7A1_HUMAN_27_537_0/4zul_un1_lig.sdf"
    sdf = open(molfile).read()
    mol = sdf_to_mol(sdf)

    ret = mol_to_sdf(mol)
    return jsonify(
        moldata=ret,
    )

@app.route("/add_methyl", methods=["POST"])
def add_methyl():
    sdf = request.form["moldata"]
    atom_index = int(request.form["atom_index"])
    mol = sdf_to_mol(sdf)

    new_index = mol.GetNumAtoms()

    mol_h = Chem.AddHs(mol, addCoords=True)

    # use rdkit to add a methyl group to the atom at index 1
    new_mol = Chem.RWMol(mol)
    Chem.SanitizeMol(new_mol)
    new_mol.AddAtom(Chem.Atom("C"))
    new_mol.AddBond(atom_index, new_index, Chem.BondType.SINGLE)

    # find the first hydogen bonded to the atom at index 1 and get its position
    new_coord = None
    for atom in mol_h.GetAtomWithIdx(atom_index).GetNeighbors():
        if atom.GetSymbol() == "H":
            new_coord = mol_h.GetConformer().GetAtomPosition(atom.GetIdx())
            break

    if new_coord is None:
        print("ERROR")
        return jsonify(
            moldata=sdf,
        )

    new_mol.GetConformer().SetAtomPosition(new_index, new_coord)

    # now convert back to a mol object and return
    new_mol.UpdatePropertyCache(strict=False)
    new_mol = new_mol.GetMol()

    # convert back to sdf and return
    moldata = mol_to_sdf(new_mol)

    print("old")
    print(sdf)
    print("new")
    print(moldata)

    return jsonify(
        moldata=moldata,
    )


if __name__ == '__main__':
    app.run(debug=True)