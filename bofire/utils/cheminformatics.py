import warnings
from typing import List

import numpy as np

try:
    from rdkit.Chem import AllChem, Descriptors, MolFromSmiles  # type: ignore
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    warnings.warn(
        "rdkit not installed, BoFire's cheminformatics utilities cannot be used."
    )

# This code is based on GAUCHE: https://github.com/leojklarner/gauche/blob/main/gauche/data_featuriser/featurisation.py


def smiles2mol(smiles: str):
    """Transforms a smiles string to an rdkit mol object.

    Args:
        smiles (str): Smiles string.

    Raises:
        ValueError: If string is not a valid smiles.

    Returns:
        rdkit.Mol: rdkit.mol object
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"{smiles} is not a valid smiles string.")
    return mol


def smiles2fingerprints(
    smiles: List[str], bond_radius: int = 5, n_bits: int = 2048
) -> np.ndarray:
    """Transforms a list of smiles to an array of morgan fingerprints.

    Args:
        smiles (List[str]): List of smiles
        bond_radius (int, optional): Bond radius to use. Defaults to 5.
        n_bits (int, optional): Number of bits. Defaults to 2048.

    Returns:
        np.ndarray: Numpy array holding the fingerprints
    """
    rdkit_mols = [smiles2mol(m) for m in smiles]
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(  # type: ignore
            mol, radius=bond_radius, nBits=n_bits
        )
        for mol in rdkit_mols
    ]

    return np.asarray(fps)


def smiles2fragments(smiles: List[str]) -> np.ndarray:
    """Transforms smiles to an array of fragments.

    Args:
        smiles (List[str]): List of smiles

    Returns:
        np.ndarray: Array holding the fragment information.
    """
    # descList[115:] contains fragment-based features only
    # (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)
    # Update: in the new RDKit version the indices are [124:]
    fragments = {d[0]: d[1] for d in Descriptors.descList[124:]}
    frags = np.zeros((len(smiles), len(fragments)))
    for i in range(len(smiles)):
        mol = smiles2mol(smiles[i])
        features = [fragments[d](mol) for d in fragments]
        frags[i, :] = features

    return frags


def smiles2bag_of_characters(smiles: List[str], max_ngram: int = 5) -> np.ndarray:
    """Transforms list of smiles to bag of characters.

    Args:
        smiles (List[str]): List of smiles
        max_ngram (int, optional): Maximal ngram value. Defaults to 5.

    Returns:
        np.ndarray: Array holding the bag of characters.
    """
    for smi in smiles:
        smiles2mol(smi)
    cv = CountVectorizer(ngram_range=(1, max_ngram), analyzer="char", lowercase=False)
    return cv.fit_transform(smiles).toarray()
