import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    root_mean_squared_error,
    r2_score, 
)
import seaborn as sns
from sklearn.utils.multiclass import unique_labels

from data_loader import config_loader
from data_loader.data_preprocessing import DataSet
from logging_util.logger import get_logger

logger = get_logger(__name__)
config = config_loader.load_config()


def get_evaluation_metrics_reg(
        true_values: pd.Series | np.ndarray, 
        predicted_values: pd.Series | np.ndarray
) -> pd.Series:
    return pd.Series(
        {
            'R2':   r2_score(true_values, predicted_values),
            'RMSE': root_mean_squared_error(true_values, predicted_values),
            'MSE':  mean_squared_error(true_values, predicted_values),
            'MAE':  mean_absolute_error(true_values, predicted_values),
        }
    )


class Eval_metrics:
    def __init__(self, num_iterations, estimators, outpath):
        self.num_it = num_iterations
        self.outpath = outpath

        self.estimator_names = (
            list(estimator.abbrev for estimator in estimators.values()) + ['naive']
        )
        self.metric_list = ['MAE', 'MSE', 'RMSE', 'R2']
        self.metrics = {
            'test': pd.DataFrame(
                columns = [i for i in range(self.num_it)],
                index = [np.array([metric for metric in self.metric_list for _ in range(len(self.estimator_names))]),
                        np.array((len(self.metric_list)) * self.estimator_names)], 
            ) , 
            'train': pd.DataFrame(
                columns = [i for i in range(self.num_it)],
                index = [np.array([metric for metric in self.metric_list for _ in range(len(self.estimator_names))]),
                        np.array((len(self.metric_list)) * self.estimator_names)], 
            )
        }

    def X_true(self, dataset, mode):
        return dataset.X_test if mode=='test' else dataset.X_train
    
    def y_true(self, dataset, mode):
        return dataset.y_test if mode=='test' else dataset.y_train
    

    def naive_metrics(self, dataset: DataSet, it: int) -> None:
        # predicting the average
        for mode in ['test', 'train']:
            y_true = self.y_true(dataset, mode)
            average_train = np.full(len(y_true), dataset.y_train.mean())
            for metric in self.metric_list:
                self.metrics[mode].loc[(metric, 'naive'), it] = get_evaluation_metrics_reg(y_true, average_train)[metric]


    def calculate_metrics(
        self, estimator, estimator_dataset: DataSet, it: int, plot_: bool=False, verbose: bool=True
        ) -> None:
        for mode in ['test', 'train']:
            metric_vals = estimator.get_evaluation_metrics(estimator_dataset, set=mode)
            for metric in self.metric_list:
                if metric in metric_vals.index:
                    self.metrics[mode].loc[(metric, estimator.abbrev), it] = metric_vals[metric]

            # add mean and std
            if self.num_it > 1:
                self.metrics[mode]['mean'] = self.metrics[mode][range(self.num_it)].mean(axis=1)
                self.metrics[mode]['std'] = self.metrics[mode][range(self.num_it)].std(axis=1)
                self.metrics[mode] = self.metrics[mode][['mean', 'std'] + [i for i in range(self.num_it)]]

            self.metrics[mode].to_csv(self.outpath / f'performance_metrics_{mode}.csv', index=True)
        
        if verbose:
            print(self.metrics['test'])
        

def plot_prediction(true_values: pd.Series | np.ndarray, predicted_values: pd.Series | np.ndarray, path: str | Path, it: int):
    plt.figure()
    plt.scatter(predicted_values, true_values, c='b', marker='o')
    plt.plot([0, max(predicted_values)], [0, max(predicted_values)], c='r', label='ideal scenario')
    
    plt.ylabel(f"measured {true_values.name}")
    plt.xlabel(f"predicted {true_values.name}")
    plt.legend()
    plt.savefig(path / f'estimation_{it}.png')
    plt.close()