import warnings
from typing import List, Optional

import numpy as np
import pandas as pd


try:
    from rdkit import RDLogger
    from rdkit.Chem import AllChem, Descriptors, MolFromSmiles

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    warnings.warn(
        "rdkit not installed, BoFire's cheminformatics utilities cannot be used.",
        ImportWarning,
    )

try:
    from mordred import Calculator, descriptors
except ImportError:
    warnings.warn(
        "mordred not installed. Mordred molecular descriptors cannot be used.",
        ImportWarning,
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
    mol = MolFromSmiles(smiles)  # type: ignore
    if mol is None:
        raise ValueError(f"{smiles} is not a valid smiles string.")
    return mol


def smiles2fingerprints(
    smiles: List[str],
    bond_radius: int = 5,
    n_bits: int = 2048,
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
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=bond_radius, nBits=n_bits)  # type: ignore
        for mol in rdkit_mols
    ]

    return np.asarray(fps)


def smiles2fragments(
    smiles: List[str],
    fragments_list: Optional[List[str]] = None,
) -> np.ndarray:
    """Transforms smiles to an array of fragments.

    Args:
        smiles (list[str]): List of smiles
        fragments_list (list[str], optional): List of desired fragments. Defaults to None.

    Returns:
        np.ndarray: Array holding the fragment information.

    """
    rdkit_fragment_list = [
        item
        for item in Descriptors.descList
        if item[0].startswith("fr_")  # type: ignore
    ]
    if fragments_list is None:
        fragments = {d[0]: d[1] for d in rdkit_fragment_list}
    else:
        fragments = {d[0]: d[1] for d in rdkit_fragment_list if d[0] in fragments_list}

    frags = np.zeros((len(smiles), len(fragments)))
    for i, smi in enumerate(smiles):
        mol = smiles2mol(smi)
        features = [fragments[d](mol) for d in fragments]
        frags[i, :] = features

    return frags


# def smiles2bag_of_characters(smiles: List[str], max_ngram: int = 5) -> np.ndarray:
#     """Transforms list of smiles to bag of characters.
#
#     Args:
#         smiles (List[str]): List of smiles
#         max_ngram (int, optional): Maximal ngram value. Defaults to 5.
#
#     Returns:
#         np.ndarray: Array holding the bag of characters.
#     """
#     for smi in smiles:
#         smiles2mol(smi)
#     cv = CountVectorizer(ngram_range=(1, max_ngram), analyzer="char", lowercase=False)
#     return cv.fit_transform(smiles).toarray()


def smiles2mordred(smiles: List[str], descriptors_list: List[str]) -> np.ndarray:
    """Transforms list of smiles to mordred moelcular descriptors.

    Args:
        smiles (List[str]): List of smiles
        descriptors_list (List[str]): List of desired mordred descriptors

    Returns:
        np.ndarray: Array holding the mordred moelcular descriptors.

    """
    mols = [smiles2mol(smi) for smi in smiles]

    calc = Calculator(descriptors, ignore_3D=True)  # type: ignore
    calc.descriptors = [d for d in calc.descriptors if str(d) in descriptors_list]

    descriptors_df = calc.pandas(mols)
    nan_list = [
        pd.to_numeric(descriptors_df[col], errors="coerce").isnull().values.any()  # type: ignore
        for col in descriptors_df.columns
    ]
    if any(nan_list):
        raise ValueError(
            f"Found NaN values in descriptors {list(descriptors_df.columns[nan_list])}",  # type: ignore
        )

    return descriptors_df.astype(float).values


def smiles2fragments_fingerprints(
    smiles: List[str],
    bond_radius: int = 5,
    n_bits: int = 2048,
    fragments_list: Optional[List[str]] = None,
) -> np.ndarray:
    fingerprints = smiles2fingerprints(smiles, bond_radius=bond_radius, n_bits=n_bits)
    fragments = smiles2fragments(smiles, fragments_list=fragments_list)

    return np.hstack((fingerprints, fragments))
