import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import partial_dependence
import shap
import warnings
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

from data_loader import config_loader
from data_loader.data_preprocessing import DataSet
from logging_util.logger import get_logger

logger = get_logger(__name__)
config = config_loader.load_config()


def split_subsets(df, steelslag_presence):
    """
    select subdataframe with columns with steel slag (steelslag_presence = 1) 
    or without steel slag (steelslag_presence = 0)
    """
    if steelslag_presence == 1:
        if 'Steelslag_3_conc' in df.columns:
            return df.loc[
                (df['Steelslag_1_conc']>0) | (df['Steelslag_0.5_conc']>0) 
                | (df['Steelslag_3_conc']>0) | (df['Steelslag_mix_conc']>0)
            ]
        elif 'Steelslag_SA' in df.columns:
            return df.loc[df['Steelslag_SA']>0]
        elif 'rel_Steelslag_SA' in df.columns:
            return df.loc[df['rel_Steelslag_SA']>0]

    else:
        if 'Steelslag_3_conc' in df.columns:
            return df.loc[
            (df['Steelslag_1_conc']==0) * (df['Steelslag_0.5_conc']==0) 
            * (df['Steelslag_3_conc']==0) * (df['Steelslag_mix_conc']==0)
        ]
        elif 'Steelslag_SA' in df.columns:
            return df.loc[df['Steelslag_SA']==0]
        elif 'rel_Steelslag_SA' in df.columns:
            return df.loc[df['rel_Steelslag_SA']==0]


# def apply_shap(estimator_cls, model_name: str, fold: int, target: str, 
#                filepath: str, resultspath: str, test: bool = True, 
#                subsample_n: int = 0, save: bool = True, per_subset=False,
#                ):
#     # load input data
#     x = pd.read_csv(filepath / 'data' / f'{"test" if test else "train"}_set_{fold}.csv', index_col=0) 
#     if subsample_n:
#         x = x.sample(n=subsample_n, random_state=42)

#     # load estimator
#     estimator = estimator_cls(outpath=filepath)
#     estimator = estimator.load(path=estimator.output_path, it=fold)

#     # set resultspath
#     resultspath.mkdir(parents=True, exist_ok=True)

#     # calculate shap values
#     if per_subset: # calculate shap values of subset with/without steel slag
#         explainer_ss = shap.KernelExplainer(estimator.model.predict, split_subsets(x, 1))
#         explainer_noss = shap.KernelExplainer(estimator.model.predict, split_subsets(x, 0))
#         explainers = [explainer_ss, explainer_noss]
#         datasets = [x.loc[x['Steelslag_SA'] > 0], x.loc[dataset_x['Steelslag_SA'] == 0]]
#     else: # calculate shap values of all data
#         explainers = [shap.KernelExplainer(estimator.model.predict, x)]
#         datasets = [x]

#     shap_results = []
#     for i in range(len(explainers)):
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore")
#             shap_values = explainers[i].shap_values(datasets[i])

#         shap_results.append([shap_values, datasets[i].index])

#     # save shap values
#     if save:
#         if len(shap_results)==1: # shap values calculated on whole dataset
#             shap_values, ix = shap_results[0]
#             shap_values_df = pd.DataFrame(shap_values, index=ix, columns=x.columns)
#             shap_values_df.to_csv(resultspath / f'shap_values_{fold}.csv')
#         else: # shap values calculated per dataset (steel slag, no steel slag)
#             for i, subset in enumerate(['ss', 'noss']):
#                 shap_values, ix = shap_results[i]
#                 shap_values_df = pd.DataFrame(shap_values, index=ix, columns=x.columns)
#                 shap_values_df.to_csv(resultspath / f'shap_values_{fold}_{subset}.csv')



def shap_vals(
        estimator, dataset_x_enc: pd.DataFrame, 
        path: Path, it: int, plot_: bool=True, per_subset=False
    ) -> tuple[np.array, pd.Index]:
    logger.debug(f'Shap values.')

    datasets = [dataset_x_enc]
    if per_subset: # calculate shap values of subset with and without steel slag
        datasets += [split_subsets(dataset_x_enc, 1), split_subsets(dataset_x_enc, 0)]
        
    results = []
    for i, dataset in enumerate(datasets):
        explainer = shap.KernelExplainer(estimator.model.predict, dataset)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            shap_values = explainer.shap_values(dataset)

        if path:
            shap_values_df = pd.DataFrame(
                shap_values, index=dataset.index, columns=dataset.columns
            )
            postfix = ['', 'ss', 'noss'][i]
            shap_values_df.to_csv(
                path / f'shap_values_{it}_{postfix}.csv'
            )
        results.append([shap_values, dataset.index])

    return results


def decode_shap(X, shap_vals_enc):
    """
    decode computed shap values by summing over the encoded variables 
    of the categorical features
    """
    shap_vals_ = shap_vals_enc.copy()
    cat_var = [var for var in X.columns if pd.api.types.is_object_dtype(X[var])]
    num_var = [var for var in X.columns if var in config.numerical_var]
    bool_var = [var for var in X.columns if var in config.boolean_var]
    other_var = [var for var in X.columns if var not in cat_var + num_var + bool_var]
    assert ~len(other_var), f'variable {other_var} not defined in config (or is categorical with wrong dtype)!'

    for var in cat_var:
        ohe = OneHotEncoder()
        tmp = ohe.fit_transform(np.array(X[[var]])).toarray()

        encoded_var = list(ohe.get_feature_names_out([var]))  # new variable names
        shap_sum = shap_vals_enc[encoded_var].sum(axis=1)
        shap_vals_ = shap_vals_.drop(encoded_var, axis=1)
        shap_vals_[var] = shap_sum
    
    return shap_vals_
