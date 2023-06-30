import os
import random

# os.chdir(r'C:\Users\S31015\python_projects\bofire\bofire')
import sys

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import bofire.data_models.acquisition_functions.api as dm_acquisition_functions
import bofire.data_models.domain.api as dm_domain
import bofire.data_models.features.api as dm_features
import bofire.data_models.kernels.api as dm_kernels
import bofire.data_models.molfeatures.api as dm_molfeatures
import bofire.data_models.objectives.api as dm_objectives
import bofire.data_models.strategies.api as dm_strategies
import bofire.data_models.surrogates.api as dm_surrogates
import bofire.strategies.api as strategies_api
import bofire.surrogates.api as surrogates_api
from bofire.data_models.enum import CategoricalEncodingEnum, MolecularEncodingEnum

# appending a path
# sys.path.append(r'C:\Users\S31015\python_projects\bofire\bofire')


project_name = "ester_molecule_prediction"

if __name__ == "__main__":
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
        "(1E,5E,9Z)-1,5,9-cyclododecatriene",
        "(1Z,5Z)-1,5-cyclooctadiene",
        "1-(1,1-dimethylethyl)-4-ethylbenzene",
        "1,2,4-triethenylcyclohexane",
        "1,4-dioxacyclohexadecane-5,16-dione",
        "2,2-bis(1-methylethyl)-1,3-dioxolane",
        "2,4-dimethyl-3-pentanamine",
        "2,4-dimethyl-3-pentanol",
    ]
    smiles = [
        # 'C1=CCCC=CCCC=CCC1',
        # 'C\\1=C\\CC/C=C\\CC/1',
        # 'CCC1=CC=C(C=C1)C(C)(C)C',
        # 'C=CC1CCC(C=C)C(C1)C=C',
        # 'O=C1OCCOC(CCCCCCCCCC1)=O',
        # 'CC(C)C1(OCCO1)C(C)C',
        # 'CC(C(C(C)C)N)C',
        # 'CC(C)C(O)C(C)C',
        "CC(=O)Oc1ccccc1C(=O)O",
        "c1ccccc1",
        "[CH3][CH2][OH]",
        # "C-C-O",
        # "OCC",
        "N[C@](C)(F)C(=O)O",
    ]
    experiments = [
        # ['C1=CCCC=CCCC=CCC1', 88.0],
        # ['C\\1=C\\CC/C=C\\CC/1', 35.0],
        # ['CCC1=CC=C(C=C1)C(C)(C)C', 69.0],
        # ['C=CC1CCC(C=C)C(C1)C=C', 69.0],
        # ['O=C1OCCOC(CCCCCCCCCC1)=O', 165.0],
        # ['CC(C)C1(OCCO1)C(C)C', 48.0],
        # ['CC(C(C(C)C)N)C', 20.0],
        # ['CC(C)C(O)C(C)C', 42.0]
        ["CC(=O)Oc1ccccc1C(=O)O", 88.0],
        ["c1ccccc1", 35.0],
        ["[CH3][CH2][OH]", 69.0],
        # ['C=CC1CCC(C=C)C(C1)C=C', 69.0],
        # ["C-C-O", 165.0],
        # ["OCC", 48.0],
        ["N[C@](C)(F)C(=O)O", 20.0],
    ]
    X_columns = ["molecule"]
    Y_columns = ["target"]

    experiments = pd.DataFrame(experiments, columns=X_columns + Y_columns)
    experiments[f"valid_target"] = 1

    # mordred_descriptors = ['NssCH2','ATSC2d']
    mordred_descriptors = ["NssCH2", "ATSC2d"]

    in1 = dm_features.MolecularInput(
        key="molecule",
        # categories=names,
        smiles=smiles,
        # molfeatures=dm_molfeatures.Fingerprints(n_bits=32),
        # molfeatures=dm_molfeatures.Fragments(),
        # molfeatures=dm_molfeatures.FingerprintsFragments(n_bits=32),
        molfeatures=dm_molfeatures.MordredDescriptors(descriptors=mordred_descriptors),
    )
    # in2 = dm_features.ContinuousInput(key='temperature', bounds=(290, 310))

    input_features = dm_domain.Inputs(features=[in1])
    output_features = dm_domain.Outputs(
        features=[
            dm_features.ContinuousOutput(
                key="target", objective=dm_objectives.MaximizeObjective(w=1.0)
            )
        ]
    )
    constraints = dm_domain.Constraints()

    domain = dm_domain.Domain(
        inputs=input_features,
        outputs=output_features,
        constraints=constraints,
    )

    surrogates_data_model = dm_surrogates.BotorchSurrogates(
        surrogates=[
            dm_surrogates.SingleTaskGPSurrogate(
                # dm_surrogates.MixedTanimotoGPSurrogate(
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
                    # 'cat_descriptor_input': CategoricalEncodingEnum.DESCRIPTOR,
                    # 'cat_input': CategoricalEncodingEnum.ONE_HOT
                },
            )
            for output_name in Y_columns
        ],
    )

    surrogates = surrogates_api.BotorchSurrogates(data_model=surrogates_data_model)
    surrogates.fit(experiments=experiments)

    # preds = surrogates.surrogates[0].predict(experiments)
    # rmse = mean_squared_error(experiments['target'], preds['target_pred'], squared=False)
    # r2 = r2_score(experiments['target'], preds['target_pred'])

    # strategy_data_model = dm_strategies.QnehviStrategy(
    strategy_data_model = dm_strategies.MultiplicativeSoboStrategy(
        domain=domain,
        acquisition_function=dm_acquisition_functions.qNEI(),
        num_raw_samples=8,
        num_restarts=2,
        num_sobol_samples=8,
        surrogate_specs=surrogates_data_model,
    )
    strategy = strategies_api.map(strategy_data_model)

    strategy.tell(experiments)

    candidates = strategy.ask(candidate_count=2)

    cv_results = strategy.surrogate_specs.surrogates[0].cross_validate(
        experiments, folds=5
    )
    cv_train = cv_results[0].get_metrics()
    cv_test = cv_results[1].get_metrics()

    for index, surrogate in enumerate(strategy.surrogate_specs.surrogates):
        dump = surrogate.dumps()
        text_file = open(
            os.path.join(
                os.getcwd(),
                "projects",
                project_name,
                f"{output_features.get_keys()[index]}_surrogate_fragprints_dump.txt",
            ),
            "wt",
        )
        n = text_file.write(dump)
        text_file.close()
        print(dump)

    # surrogates_list = surrogates.compatibilize(input_features, output_features)

    print()
