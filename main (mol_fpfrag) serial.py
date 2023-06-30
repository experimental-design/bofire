import os

# os.chdir(r'C:\Users\S31015\python_projects\bofire\bofire')
import sys

# appending a path
# sys.path.append(r'C:\Users\S31015\python_projects\bofire\bofire')

import bofire.data_models.domain.api as dm_domain
import bofire.data_models.features.api as dm_features
import bofire.data_models.surrogates.api as dm_surrogates
import bofire.data_models.strategies.api as dm_strategies
import bofire.data_models.acquisition_functions.api as dm_acquisition_functions
import bofire.data_models.objectives.api as dm_objectives
import bofire.data_models.molfeatures.api as dm_molfeatures
import bofire.data_models.kernels.api as dm_kernels

import bofire.surrogates.api as surrogates_api
import bofire.strategies.api as strategies_api

from bofire.data_models.enum import MolecularEncodingEnum, CategoricalEncodingEnum

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import random

from pydantic import parse_obj_as
import json


project_name = "ester_molecule_prediction"

if __name__ == "__main__":
    # XY_data_df['exp_fp'] = XY_data_df['exp_fp'].astype(float)
    names = [
        "(1E,5E,9Z)-1,5,9-cyclododecatriene",
        "(1Z,5Z)-1,5-cyclooctadiene",
        "1-(1,1-dimethylethyl)-4-ethylbenzene",
        "1,2,4-triethenylcyclohexane",
        # '1,4-dioxacyclohexadecane-5,16-dione',
        # '2,2-bis(1-methylethyl)-1,3-dioxolane',
        # '2,4-dimethyl-3-pentanamine',
        # '2,4-dimethyl-3-pentanol',
    ]
    smiles = [
        "C1=CCCC=CCCC=CCC1",
        "C\\1=C\\CC/C=C\\CC/1",
        "CCC1=CC=C(C=C1)C(C)(C)C",
        "C=CC1CCC(C=C)C(C1)C=C",
        # 'O=C1OCCOC(CCCCCCCCCC1)=O',
        # 'CC(C)C1(OCCO1)C(C)C',
        # 'CC(C(C(C)C)N)C',
        # 'CC(C)C(O)C(C)C',
    ]
    experiments = [
        ["C1=CCCC=CCCC=CCC1", 88.0],
        ["C\\1=C\\CC/C=C\\CC/1", 35.0],
        ["CCC1=CC=C(C=C1)C(C)(C)C", 69.0],
        ["C=CC1CCC(C=C)C(C1)C=C", 69.0],
        # ['1,4-dioxacyclohexadecane-5,16-dione', 298.15, 'B', 165.0],
        # ['2,2-bis(1-methylethyl)-1,3-dioxolane', 298.15, 'A', 48.0],
        # ['2,4-dimethyl-3-pentanamine', 298.15, 'B', 20.0],
        # ['2,4-dimethyl-3-pentanol', 298.15, 'A', 42.0]
    ]
    X_columns = ["molecule"]
    Y_columns = ["target"]

    experiments = pd.DataFrame(experiments, columns=X_columns + Y_columns)
    experiments[f"valid_target"] = 1

    mordred_descriptors = ["NssCH2", "ATSC2d"]

    in1 = dm_features.MolecularInput(
        key="molecule",
        # categories=names,
        smiles=smiles,
        # molfeatures=dm_molfeatures.Fingerprints(),
        # molfeatures=dm_molfeatures.Fragments(),
        # molfeatures=dm_molfeatures.FingerprintsFragments(),
        molfeatures=dm_molfeatures.MordredDescriptors(descriptors=mordred_descriptors),
    )
    # in1 = dm_features.MolecularInput('molecule', smiles, dm_molfeatures.FingerprintsFragments())

    # in2 = dm_features.ContinuousInput(key='temperature', bounds=(290, 310))

    input_features = dm_domain.Inputs(features=[in1])

    input_jspec = input_features.json()
    input_loaded_jspec = parse_obj_as(dm_domain.Inputs, json.loads(input_jspec))

    output_features = dm_domain.Outputs(
        features=[
            dm_features.ContinuousOutput(
                key="target", objective=dm_objectives.MaximizeObjective(w=1.0)
            )
        ]
    )

    output_jspec = output_features.json()
    output_loaded_jspec = parse_obj_as(dm_domain.Outputs, json.loads(output_jspec))

    # constraints = dm_domain.Constraints()

    # domain = dm_domain.Domain(
    #     inputs=input_features,
    #     outputs=output_features,
    #     constraints=constraints,
    # )

    surrogate_data = dm_surrogates.SingleTaskGPSurrogate(
        inputs=input_features,
        outputs=output_features,
        # scaler=dm_surrogates.ScalerEnum.NORMALIZE,
        scaler=dm_surrogates.ScalerEnum.IDENTITY,
        kernel=dm_kernels.ScaleKernel(base_kernel=dm_kernels.TanimotoKernel()),
        input_preprocessing_specs={
            # 'molecule': MolecularEncodingEnum.FINGERPRINTS,
            # 'molecule': MolecularEncodingEnum.FRAGMENTS,
            # 'molecule': MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS,
            "molecule": MolecularEncodingEnum.MOL_DESCRIPTOR,
        },
    )

    jspec = surrogate_data.json()

    surrogate_data = parse_obj_as(dm_surrogates.AnySurrogate, json.loads(jspec))

    surrogate = surrogates_api.map(surrogate_data)
    surrogate.fit(experiments=experiments)

    dump = surrogate.dumps()

    # predict with it
    df_predictions = surrogate.predict(experiments)
    # transform to spec
    predictions = surrogate.to_predictions(predictions=df_predictions)

    surrogate_data = parse_obj_as(dm_surrogates.AnySurrogate, json.loads(jspec))
    surrogate2 = surrogates_api.map(surrogate_data)
    surrogate2.loads(dump)

    # predict with it
    df_predictions2 = surrogate2.predict(experiments)
    # transform to spec
    predictions2 = surrogate2.to_predictions(predictions=df_predictions2)

    # check for equality
    predictions == predictions2
