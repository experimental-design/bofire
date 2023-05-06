import copy
import torch
import base64
import io
import warnings
import os
os.chdir(r'C:\Users\S31015\python_projects\bofire_test\bofire')
import sys
# appending a path
sys.path.append(r'C:\Users\S31015\python_projects\bofire_test\bofire')

import bofire.data_models.domain.api as dm_domain
import bofire.data_models.features.api as dm_features
import bofire.data_models.surrogates.api as dm_surrogates

import bofire.surrogates.api as surrogates_api

from bofire.data_models.enum import MolecularEncodingEnum

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

project_name = 'ester_molecule_prediction'

if __name__ == '__main__':
    # Import data for ML model fitting
    external_data_df = pd.read_pickle(os.path.join(os.getcwd(),'projects', project_name, 'External_Flash point.pkl'))[['SMILES', 'temperature','exp_fp']]
    oa_data_df = pd.read_pickle(os.path.join(os.getcwd(),'projects', project_name, 'OA_Flash point.pkl'))[['SMILES', 'temperature','exp_fp']]

    XY_data_df = pd.concat((external_data_df, oa_data_df), axis=0).reset_index(drop=True)
    XY_data_df['exp_fp'] = XY_data_df['exp_fp'].astype(float)

    X_columns =['SMILES']
    Y_columns = ['exp_fp']

    experiments = XY_data_df[X_columns + Y_columns]
    # experiments = experiments.drop(
    #     [experiments[experiments['SMILES'] == 'C(=O)N'].index[0], experiments[experiments['SMILES'] == 'CO'].index[0]],
    #     axis=0).reset_index(drop=True)

    experiments, experiments_test = train_test_split(experiments, test_size=0.2, random_state=42)

    for y in Y_columns:
        experiments[f"valid_{y}"] = 1

    # fp_descriptors = ['NssCH2', 'VE1_A', 'nHetero', 'AATS1d', 'ATSC2d', 'GATS4d', 'SIC2', 'SlogP_VSA3', 'EState_VSA3', 'EState_VSA4', 'piPC5', 'piPC10']

    # in1 = dm_features.CategoricalMolecularDescriptorInput(key='SMILES', descriptors=['SpAbs_A','SpMax_A'])
    # in1 = dm_features.CategoricalMolecularDescriptorInput(key='SMILES', descriptors=fp_descriptors)
    in1 = dm_features.MolecularInput(key='SMILES')
    # in2 = dm_features.ContinuousInput(key='temperature', bounds=(0, 100))

    # input_features = dm_domain.Inputs(features=[in1, in2])
    input_features = dm_domain.Inputs(features=[in1])
    output_features = dm_domain.Outputs(features=[dm_features.ContinuousOutput(key=output_name) for output_name in Y_columns])
    # constraints = dm_domain.Constraints()

    # domain = dm_domain.Domain(
    #     input_features=input_features,
    #     output_features=output_features,
    #     constraints=constraints,
    # )

    surrogates_data_model = dm_surrogates.BotorchSurrogates(
        surrogates=[
            dm_surrogates.TanimotoGPSurrogate(
                input_features=input_features,
                output_features=dm_domain.Outputs(
                    features=[dm_features.ContinuousOutput(key=output_name)]
                ),
                # scaler=dm_surrogates.ScalerEnum.NORMALIZE,
    input_preprocessing_specs={'SMILES': MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS},
    # input_preprocessing_specs={'SMILES': CategoricalMolecularEncodingEnum.FINGERPRINTS},
    # input_preprocessing_specs={'SMILES': CategoricalMolecularEncodingEnum.FRAGMENTS},
            ) for output_name in Y_columns
        ],
    )

    surrogates = surrogates_api.BotorchSurrogates(data_model=copy.copy(surrogates_data_model))
    surrogates.fit(experiments=experiments)

    # cv_results = surrogates.surrogates[0].cross_validate(experiments, folds=5)
    # cv_train = cv_results[0].get_metrics()
    # cv_test = cv_results[1].get_metrics()

    preds_train = surrogates.surrogates[0].predict(experiments)
    rmse_train = mean_squared_error(experiments['exp_fp'], preds_train['exp_fp_pred'], squared=False)
    r2_train = r2_score(experiments['exp_fp'], preds_train['exp_fp_pred'])
    preds_test = surrogates.surrogates[0].predict(experiments_test)
    rmse_test = mean_squared_error(experiments_test['exp_fp'], preds_test['exp_fp_pred'], squared=False)
    r2_test = r2_score(experiments_test['exp_fp'], preds_test['exp_fp_pred'])

    for index, surrogate in enumerate(surrogates.surrogates):
        dump = surrogate.dumps()
        text_file = open(os.path.join(os.getcwd(),'projects', project_name, 'exp_fp_surrogate_fragprints_dump.txt'), "wt")
        n = text_file.write(dump)
        text_file.close()
        print(dump)

    def loads(str):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import bofire.surrogates.cloudpickle_module as cloudpickle_module

            if len(w) == 1:
                raise ModuleNotFoundError("Cloudpickle is not available.")

        buffer = io.BytesIO(base64.b64decode(str.encode()))
        return torch.load(buffer, pickle_module=cloudpickle_module)  # type: ignore

    surrogates_load = surrogates_api.BotorchSurrogates(data_model=copy.copy(surrogates_data_model))

    surrogates_load.surrogates[0].model = loads(
        open(os.path.join(os.getcwd(), 'projects', project_name, 'exp_fp_surrogate_fragprints_dump.txt'), "r").read())

    preds_train_load = surrogates_load.surrogates[0].predict(experiments)
    rmse_train_load = mean_squared_error(experiments['exp_fp'], preds_train_load['exp_fp_pred'], squared=False)
    r2_train_load = r2_score(experiments['exp_fp'], preds_train_load['exp_fp_pred'])
    preds_test_load = surrogates_load.surrogates[0].predict(experiments_test)
    rmse_test_load = mean_squared_error(experiments_test['exp_fp'], preds_test_load['exp_fp_pred'], squared=False)
    r2_test_load = r2_score(experiments_test['exp_fp'], preds_test_load['exp_fp_pred'])


    print()
