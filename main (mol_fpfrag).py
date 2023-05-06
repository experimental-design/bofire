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

import bofire.surrogates.api as surrogates_api
import bofire.strategies.api as strategies_api

from bofire.data_models.enum import MolecularEncodingEnum, CategoricalEncodingEnum

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import random


project_name = 'ester_molecule_prediction'

if __name__ == '__main__':
    # Import data for ML model fitting
    # external_data_df = pd.read_pickle(os.path.join(os.getcwd(),'projects', project_name, 'External_Flash point.pkl'))[['SMILES', 'temperature','exp_fp']]
    # XY_data_df = pd.read_pickle('OA_Flash point.pkl')[['SMILES', 'temperature','exp_fp']]
    #
    # # XY_data_df = pd.concat((external_data_df, oa_data_df), axis=0).reset_index(drop=True)
    #
    # XY_data_df['cat_descriptor_input'] = pd.Series(
    #     random.choices(["A", "B", "C"], weights=[1, 1, 1], k=len(XY_data_df)),
    #     index=XY_data_df.index
    # )
    #
    # XY_data_df['exp_fp'] = XY_data_df['exp_fp'].astype(float)
    names = [
        '(1E,5E,9Z)-1,5,9-cyclododecatriene',
        '(1Z,5Z)-1,5-cyclooctadiene',
        '1-(1,1-dimethylethyl)-4-ethylbenzene',
        '1,2,4-triethenylcyclohexane',
        # '1,4-dioxacyclohexadecane-5,16-dione',
        # '2,2-bis(1-methylethyl)-1,3-dioxolane',
        # '2,4-dimethyl-3-pentanamine',
        # '2,4-dimethyl-3-pentanol',
    ]
    smiles = [
        'C1=CCCC=CCCC=CCC1',
        'C\\1=C\\CC/C=C\\CC/1',
        'CCC1=CC=C(C=C1)C(C)(C)C',
        'C=CC1CCC(C=C)C(C1)C=C',
        # 'O=C1OCCOC(CCCCCCCCCC1)=O',
        # 'CC(C)C1(OCCO1)C(C)C',
        # 'CC(C(C(C)C)N)C',
        # 'CC(C)C(O)C(C)C',
    ]
    experiments = [
        ['(1E,5E,9Z)-1,5,9-cyclododecatriene', 88.0],
        ['(1Z,5Z)-1,5-cyclooctadiene', 35.0],
        ['1-(1,1-dimethylethyl)-4-ethylbenzene', 69.0],
        ['1,2,4-triethenylcyclohexane', 69.0],
        # ['1,4-dioxacyclohexadecane-5,16-dione', 298.15, 'B', 165.0],
        # ['2,2-bis(1-methylethyl)-1,3-dioxolane', 298.15, 'A', 48.0],
        # ['2,4-dimethyl-3-pentanamine', 298.15, 'B', 20.0],
        # ['2,4-dimethyl-3-pentanol', 298.15, 'A', 42.0]
    ]
    X_columns = ['molecule']
    Y_columns = ['target']

    experiments = pd.DataFrame(experiments, columns=X_columns + Y_columns)
    experiments[f"valid_target"] = 1

    mordred_descriptors = ['NssCH2','ATSC2d']

    in1 = dm_features.MolecularInput(
        key='molecule',
        categories=names,
        smiles=smiles,
        # molfeatures=dm_molfeatures.Fingerprints(),
        # molfeatures=dm_molfeatures.Fragments(),
        molfeatures=dm_molfeatures.FingerprintsFragments(),
        # molfeatures=dm_molfeatures.MordredDescriptors(descriptors=mordred_descriptors),
    )

    input_features = dm_domain.Inputs(features=[in1])
    output_features = dm_domain.Outputs(features=[dm_features.ContinuousOutput(key='target', objective=dm_objectives.MaximizeObjective(w=1.0))])
    constraints = dm_domain.Constraints()

    domain = dm_domain.Domain(
        inputs=input_features,
        outputs=output_features,
        constraints=constraints,
    )

    surrogates_data_model = dm_surrogates.BotorchSurrogates(
        surrogates=[
            # dm_surrogates.SingleTaskGPSurrogate(
            dm_surrogates.MixedTanimotoGPSurrogate(
                    inputs=input_features,
                outputs=dm_domain.Outputs(
                    features=[dm_features.ContinuousOutput(key=output_name)]
                ),
                scaler=dm_surrogates.ScalerEnum.NORMALIZE,
    input_preprocessing_specs={
        # 'molecule': MolecularEncodingEnum.FINGERPRINTS,
        # 'molecule': MolecularEncodingEnum.FRAGMENTS,
        'molecule': MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS,
        # 'molecule': MolecularEncodingEnum.MOL_DESCRIPTOR,
        # 'cat_descriptor_input': CategoricalEncodingEnum.DESCRIPTOR,
        # 'cat_input': CategoricalEncodingEnum.ONE_HOT
    }
                ,
            ) for output_name in Y_columns
        ],
    )

    # surrogates = surrogates_api.BotorchSurrogates(data_model=surrogates_data_model)
    # surrogates.fit(experiments=experiments)
    #
    # preds = surrogates.surrogates[0].predict(experiments)
    # rmse = mean_squared_error(experiments['exp_fp'], preds['exp_fp_pred'], squared=False)
    # r2 = r2_score(experiments['exp_fp'], preds['exp_fp_pred'])

    # strategy_data_model = dm_strategies.QnehviStrategy(
    strategy_data_model = dm_strategies.MultiplicativeSoboStrategy(
        domain=domain,
        acquisition_function=dm_acquisition_functions.qNEI(),
        num_raw_samples=8,
        num_restarts=2,
        num_sobol_samples=8,
        surrogate_specs=surrogates_data_model
    )
    strategy = strategies_api.map(strategy_data_model)

    strategy.tell(experiments)

    candidates = strategy.ask(candidate_count=2)

    cv_results = surrogates.surrogates[0].cross_validate(experiments, folds=5)
    cv_train = cv_results[0].get_metrics()
    cv_test = cv_results[1].get_metrics()

    for index, surrogate in enumerate(surrogates.surrogates):
        dump = surrogate.dumps()
        text_file = open(os.path.join(os.getcwd(),'projects', project_name, f"{output_features.get_keys()[index]}_surrogate_fragprints_dump.txt"), "wt")
        n = text_file.write(dump)
        text_file.close()
        print(dump)

    surrogates_list = surrogates.compatibilize(input_features, output_features)

    print()
