from __future__ import annotations
import pathlib
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel, parse_obj_as
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, LeaveOneOut, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from data_loader.config_loader import load_config
from logging_util.logger import get_logger

config = load_config()
logger = get_logger(__name__)


class DataSet(BaseModel):
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_valid: pd.DataFrame | None
    y_valid: pd.Series | None

    class Config:
        arbitrary_types_allowed = True

    def save(self, path: Path) -> None:
        joblib.dump(self.__dict__, path / 'dataset.pkl')

    @classmethod
    def load(self, path: Path) -> DataSet:
        return parse_obj_as(DataSet, joblib.load(path / 'dataset.pkl'))


def check_repeats(df: pd.DataFrame, repeated: pd.Series):
    """
    add column filled in which repeated columns get the same value
    """
    df['repeat_indications'] = range(len(df))
    for i in df.index:
        repeats = eval(repeated[i])
        if len(repeats):
            for rep in repeats:
                if rep in df.index:
                    df.loc[rep, 'repeat_indications'] = df.loc[i, 'repeat_indications']

    return df


def stratifiedKfold_regr(
        df: pd.DataFrame, k: int, features: list, random_state, 
        repeated: pd.Series, nbins=30
    ) -> list:
    """
    Split data into k folds
    - with stratified split based on target value
    - ensuring that repeated combinations are in same fold 
    """
    df = check_repeats(df, repeated) 

    # stratification: bin based on target values
    df['bin'] = pd.qcut(
            df[config.target].astype('float64'), nbins, 
            labels=False, duplicates='drop'
    )
    cv_outer = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state)

    folds = list(cv_outer.split(
        df[features], df['bin'], groups=df['repeat_indications']
    ))

    return folds


def cv_split(
          df: pd.DataFrame,
          validation_method: str, 
          k: int=None, 
          stratified_split: bool=False,
          seed: int=None,
          nbins: int=30,
    ):
    """
    split data into k sets
    """

    # ensure that repeated batches go in the same fold
    features = config.numerical_var + config.categorical_var + config.boolean_var
    
    if validation_method == 'kfoldcv' and stratified_split:
        splits = stratifiedKfold_regr(
            df, k, features, random_state=seed, repeated=df['usable_repeats'], nbins=nbins
        )

    elif validation_method == 'kfoldcv' and not stratified_split:
        cv_outer = KFold(n_splits=k, shuffle=True, random_state=seed)
        splits = cv_outer.split(df[features])

    elif validation_method == 'loocv':
        loo = LeaveOneOut()
        splits = loo.split(df)

    else: # train test split
        splits = []
        for _ in range(k):
            X_train, X_test, y_train, y_test = train_test_split(
                df[features], 
                df[config.target], 
                test_size=config.test_size, 
                random_state=None
            )
            splits.append([
                [df.index.get_loc(ind) for ind in X_train.index], 
                [df.index.get_loc(ind) for ind in X_test.index], 
            ])
    
    return splits


def train_test_val_split(
    data: pd.DataFrame,
    outpath: pathlib.Path | None = None,
    it: int = 1,
    train_only: bool = False,
    train_i = None, 
    test_i = None,
    add_validation = False,
    random_state: int = None,
) -> DataSet:
    
    # logger.debug("Splitting data in train, test and validation set.")

    features = config.numerical_var + config.categorical_var + config.boolean_var
    if train_only:
        return DataSet(
            X_train=data[features],
            y_train=data[config.target],
            X_test=data[features],
            y_test=data[config.target],
            X_valid=None,
            y_valid=None,
        )
    
    
    # split data
    if train_i is not None: # split according to given indices (train_i, test_i)
        X_train = data.iloc[train_i][features]
        y_train = data.iloc[train_i][config.target]
        X_test = data.iloc[test_i][features]
        y_test = data.iloc[test_i][config.target]

        if add_validation:
            # get random validation set from training set
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, 
                y_train, 
                test_size=config.validation_size / (1 - config.test_size),
                random_state=random_state,
            )
        else:
            X_valid = None
            y_valid = None         

    else: # random train_test split
        X_train, X_test, y_train, y_test = train_test_split(
            data[features], data[config.target], test_size=config.test_size, random_state=random_state
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, 
            y_train, 
            test_size=config.validation_size / (1 - config.test_size),
            random_state=random_state,
        )

    # save to csv
    if outpath is not None:
        (outpath / 'data').mkdir(parents=True, exist_ok=True)
        pd.concat([X_train, y_train], axis=1).to_csv(
            outpath / 'data' / f'training_set_{it}.csv', index=True
        )
        pd.concat([X_test, y_test], axis=1).to_csv(
            outpath / 'data' / f'test_set_{it}.csv', index=True
        )
        if isinstance(X_valid, pd.DataFrame):
            pd.concat([X_valid, y_valid], axis=1).to_csv(
                outpath / 'data' / f'validation_set_{it}.csv', index=True
        )

    return DataSet(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_valid=X_valid,
        y_valid=y_valid,
    )


def one_hot_encoding(data: pd.DataFrame, dataset: pd.DataFrame = None) -> pd.DataFrame:
    """
    onehot encode categorical variables
    """
    cat_var = [var for var in data.columns if var in config.categorical_var] 
    num_var = [var for var in data.columns if var in config.numerical_var]
    bool_var = [var for var in data.columns if var in config.boolean_var]
    other_var = [var for var in data.columns if var not in cat_var + num_var + bool_var]
    assert ~len(other_var), f'variable {other_var} not defined in config!'

    if len(cat_var):
        if dataset:
            all_data = pd.concat([dataset.X_valid, dataset.X_test, dataset.X_train])
        else:
            all_data = data

        ohe = OneHotEncoder(categories=[list(all_data[var].unique()) for var in cat_var])
        tmp = ohe.fit_transform(data[cat_var]).toarray()

        encoded_var = list(ohe.get_feature_names_out(cat_var)) 
        cat_df = pd.DataFrame(tmp, columns=encoded_var, index=data.index)

        # join encoded variables to original dataframe
        data_ohe = pd.concat([data[num_var + bool_var], cat_df], axis=1)

        return data_ohe
    
    return data[num_var + bool_var]


def scaler(datasets: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    standardize numerical features
    """
    
    # indicate which var to rescale
    encoded_columns = [
        var for var in [
            'Steelslag_0.1_conc', 'Steelslag_0.5_conc', 'Steelslag_1.0_conc', 'Steelslag_3.0_conc', 
            'Diabase_3.0_conc', 'Diabase_0.063_conc', 
            'B(B)_0.063_conc', 'B(B)_3.0_conc', 'B(L)_0.063_conc', 'B(L)_2.0_conc', 
            'Dunite_0.063_conc', 'Dunite_0.125_conc', 'Dunite_1.0_conc',
            'B(B)_conc', 'B(L)_conc', 'Dunite_conc', 'Steelslag_conc', 'Diabase_conc',        
        ] if var in config.numerical_var
    ] 
    var_to_minmaxscale = [
        var for var in [
            'B(B)_SA', 'B(L)_SA', 'Dunite_SA', 'Diabase_SA', 'Steelslag_SA', 'total_SA',
            'aimed_biochar_mass', 'active_worm_nbr', 
            'worm_nbr', 'chlorotica_ratio', 'dead_worm_nbr', 'dead_chlorotica_ratio',
            'phosphorus', 'ammonium_chloride', 'urea', 
            'citric_acid_conc', 'EDTA_conc', 'Oxalate_conc', 
            'laccase_conc', 'urease_conc', 'anhydrase_conc', 
            'inoculum_density_K_petricola', 'inoculum_density_A_pullulans', 
            'inoculum_density_S_variegatus', 
            'inoculum_density_B_subtilis', 'inoculum_density_C_metallidurans', 
        ] if var in config.numerical_var
    ]
    var_to_standardize = [
        var for var in config.numerical_var if var not in (
            encoded_columns + var_to_minmaxscale
        )
    ]
   

    
    # fit scalers on training set
    sc = StandardScaler()
    scmm = MinMaxScaler() 
    sc.fit(datasets[0][var_to_standardize])
    scmm.fit(datasets[0][var_to_minmaxscale])

    # transform datasets
    datasets_rescaled = []
    for df in datasets:
        if df is not None:

            # Min max scaling for zero-meaningful features
            df_minmaxscaled = pd.DataFrame(
                scmm.transform(df[var_to_minmaxscale]), index=df.index, columns=var_to_minmaxscale
            )
            # standardizing
            df_standardized = pd.DataFrame(
                sc.transform(df[var_to_standardize]), index=df.index, columns=var_to_standardize
            )
            # paste everyting back together
            non_scaled_var = [var for var in df.columns if var not in config.numerical_var] + encoded_columns
            datasets_rescaled.append(
                pd.concat([df[non_scaled_var], df_standardized, df_minmaxscaled], axis=1)
            )
        else:
            datasets_rescaled.append(None)

    return datasets_rescaled
