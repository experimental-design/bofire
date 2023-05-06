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
        '1,4-dioxacyclohexadecane-5,16-dione',
        '2,2-bis(1-methylethyl)-1,3-dioxolane',
        '2,4-dimethyl-3-pentanamine',
        '2,4-dimethyl-3-pentanol',
    ]
    smiles = [
        'C1=CCCC=CCCC=CCC1',
        'C\\1=C\\CC/C=C\\CC/1',
        'CCC1=CC=C(C=C1)C(C)(C)C',
        'C=CC1CCC(C=C)C(C1)C=C',
        'O=C1OCCOC(CCCCCCCCCC1)=O',
        'CC(C)C1(OCCO1)C(C)C',
        'CC(C(C(C)C)N)C',
        'CC(C)C(O)C(C)C',
    ]
    experiments = [
        ['(1E,5E,9Z)-1,5,9-cyclododecatriene', 298.15, 'B', 'Y', 88.0, 30.0],
        ['(1Z,5Z)-1,5-cyclooctadiene', 298.15, 'A', 'Z', 35.0, 90.0],
        ['1-(1,1-dimethylethyl)-4-ethylbenzene', 298.15, 'C', 'Y', 69.0, 50.0],
        ['1,2,4-triethenylcyclohexane', 298.15, 'B', 'Y', 69.0, 60.0],
        # ['1,4-dioxacyclohexadecane-5,16-dione', 298.15, 'B', 165.0],
        # ['2,2-bis(1-methylethyl)-1,3-dioxolane', 298.15, 'A', 48.0],
        # ['2,4-dimethyl-3-pentanamine', 298.15, 'B', 20.0],
        # ['2,4-dimethyl-3-pentanol', 298.15, 'A', 42.0]
    ]
    X_columns = ['molecule', 'temperature', 'cat_descriptor_input', 'cat_input']
    Y_columns = ['target1', 'target2']

    experiments = pd.DataFrame(experiments, columns=X_columns + Y_columns)
    experiments[f"valid_target1"] = 1
    experiments[f"valid_target2"] = 1

    mordred_descriptors = ['NssCH2','ATSC2d']

    # in1 = dm_features.CategoricalMolecularDescriptorInput(key='SMILES', descriptors=['SpAbs_A','SpMax_A'])
    in1 = dm_features.MolecularInput(
        key='molecule',
        categories=names,
        smiles=smiles,
        # molfeatures=dm_molfeatures.Fingerprints(),
        # molfeatures=dm_molfeatures.Fragments(),
        molfeatures=dm_molfeatures.FingerprintsFragments(),
        # molfeatures=dm_molfeatures.MordredDescriptors(descriptors=mordred_descriptors),
    )
    in2 = dm_features.ContinuousInput(key='temperature', bounds=(290, 310))
    in3 = dm_features.CategoricalDescriptorInput(
        key="cat_descriptor_input",
        categories=["A", "B", "C"],
        descriptors=["d1", "d2"],
        values=[[1, 2], [3, 4], [5, 6]],
    )
    in4 = dm_features.CategoricalInput(
        key="cat_input",
        categories=["Y", "Z"],
    )

    input_features = dm_domain.Inputs(features=[in1, in2, in3, in4])
    output_features = dm_domain.Outputs(features=[dm_features.ContinuousOutput(key=y, objective=dm_objectives.MaximizeObjective(w=1.0)) for y in Y_columns])
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
        'cat_descriptor_input': CategoricalEncodingEnum.DESCRIPTOR,
        'cat_input': CategoricalEncodingEnum.ONE_HOT},
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
