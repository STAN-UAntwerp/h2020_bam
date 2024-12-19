from __future__ import annotations
import pandas as pd
import joblib
from pydantic import BaseModel, parse_obj_as
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from sklearn.model_selection import *

from data_loader.data_preprocessing import DataSet, one_hot_encoding, scaler, stratifiedKfold_regr
from data_loader.config_loader import load_config
from evaluation.evaluation import get_evaluation_metrics_reg, plot_prediction
from evaluation.feature_importance import shap_vals
from logging_util.logger import get_logger
from models.base import Estimator

logger = get_logger(__name__)
data_config = load_config()


class ParamGrid(BaseModel):
    max_depth: list[int]
    n_estimators: list[int]
    min_samples_split: list[int]
    random_state: list[int]


class Config(BaseModel):
    param_grid: ParamGrid
    cv: int
    n_jobs: int
    verbose: bool


class RandomForestEstimator(Estimator):
    configcls = Config
    abbrev = 'RF'

    def preprocess(self, dataset: DataSet) -> DataSet:
        logger.debug(f"Preprocessing data.")

        # Random forest regressor is trained with CV, so we can combine train and validation set
        X_train = pd.concat([dataset.X_train, dataset.X_valid], axis=0)
        y_train = pd.concat([dataset.y_train, dataset.y_valid], axis=0)

        # one hot encoding and standardizing
        X_train = one_hot_encoding(X_train, dataset)
        X_test = one_hot_encoding(dataset.X_test, dataset)

        
        # fill in nan values
        for var in ['chlorotica_ratio', 'dead_chlorotica_ratio']:
            X_test.loc[pd.isnull(X_test[var]), var] = 0.5
            X_train.loc[pd.isnull(X_train[var]), var] = 0.5
            
        X_train, X_test = scaler([X_train, X_test])
        self.feature_names = X_train.columns

        return DataSet(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=dataset.y_test,
            X_valid=None,
            y_valid=None,
        )


    def fit(
            self, dataset: DataSet, validation_method: str='kfoldcv', stratified=False,
            df: pd.DataFrame=None
        ) -> RandomForestEstimator:
        logger.debug(f"Training model.")
        hyperparam = self.config.dict()

        if self.cv:
            hyperparam['cv'] = self.cv  

        if validation_method == 'loocv':
            hyperparam.pop('cv')
            self.grid = GridSearchCV(
                RandomForestRegressor(), **hyperparam, scoring='neg_mean_squared_error',
                cv=LeaveOneOut()
            ) 
        else:
            if stratified:
                hyperparam.pop('cv')
                folds = stratifiedKfold_regr(
                    pd.concat([dataset.X_train, dataset.y_train.to_frame(name=data_config.target)], axis=1),
                    self.cv, self.feature_names, random_state=None,
                    repeated=df.loc[dataset.X_train.index, 'usable_repeats'], nbins=50, 
                )
                self.grid = GridSearchCV(
                    RandomForestRegressor(), **hyperparam, scoring='neg_mean_squared_error',
                    cv=list(folds)
                ) 
            else:
                self.grid = GridSearchCV(
                    RandomForestRegressor(), **hyperparam, scoring='neg_mean_squared_error'
                )
        self.grid.fit(dataset.X_train, dataset.y_train)
        self.model = self.grid.best_estimator_
        self.inner_mae = -1 * self.grid.best_score_
        self.y_pred = pd.Series(
            self.model.predict(dataset.X_test),
            index=dataset.X_test.index,
            name='predicted_' + data_config.target,
        )
        return self
    

    def get_feature_importance(self, dataset_enc: pd.DataFrame, 
                               it: int=0, plot_: bool=True, shap_: bool=False, 
                               per_subset=False, output_path=None) -> tuple:
        if not output_path:
            output_path = self.output_path        
        
        shap_vals_, shap_ix = None, None
        if shap_:
            shap_results = shap_vals(self, dataset_enc.X_test, 
                                     output_path, it, plot_, per_subset)
            shap_vals_, shap_ix = shap_results[0] # results calculated on whole dataset

        return shap_vals_, shap_ix

    
    def get_evaluation_metrics(self, dataset: DataSet, set: str, i=[]) -> pd.Series:
        if len(i) > 0:
            if set=='test':
                y = dataset.y_test.loc[i]
                y_pred = self.y_pred.loc[i]
            elif set=='train':
                y = dataset.y_train.loc[i]
                y_pred = pd.Series(self.model.predict(dataset.X_train.loc[i]), index=i, 
                                        name='predicted_' + data_config.target)
        else:
            if set=='test':
                y = dataset.y_test
                y_pred = self.y_pred
            elif set=='train':
                y = dataset.y_train
                y_pred = pd.Series(self.model.predict(dataset.X_train), index=dataset.X_train.index,
                                        name='predicted_' + data_config.target)
        return get_evaluation_metrics_reg(y, y_pred).rename('randomforest')


    def plot_prediction(self, dataset: DataSet, df: pd.DataFrame, it: int=0) -> None:
        plot_prediction(dataset.y_test, self.y_pred, self.output_path, it)


    def save(self, it: int=0) -> None: 
        # pickle grid, model, y_pred and config
        joblib.dump(
            {
                'model': self.model,
                'grid': self.grid,
                'y_pred': self.y_pred,
                'config': self.config.__dict__,
            },
            self.output_path / f'model_{it}.pkl',
        )

    @classmethod
    def load(cls, path: str | Path, it: int=0) -> RandomForestEstimator:
        model = joblib.load(path / f'model_{it}.pkl')
        estimator = RandomForestEstimator(path)

        # overwrite attributes
        estimator.config = parse_obj_as(cls.configcls, model['config'])
        estimator.model = model['model']
        estimator.grid = model['grid']
        estimator.y_pred = model['y_pred']
        return estimator
