from importlib import import_module

from data_loader import config_loader
from logging_util.logger import get_logger
from models.base import Estimator

config = config_loader.load_config()
logger = get_logger(__name__)

enabled_estimators = [
    #'LassoRegressor.estimator.LassoEstimator',
    'RandomForestRegressor.estimator.RandomForestEstimator',
    #'XgboostRegressor.estimator.XGBoostEstimator',
]

estimators: dict[str, Estimator] = {}
for estimator in enabled_estimators:
    estimator_name = estimator.split(".", maxsplit=1)[0]
    logger.info(f'Loading {estimator_name}')
    estimator_module_name, estimator_class = estimator.rsplit(".", maxsplit=1)
    estimator_module = import_module(f".{estimator_module_name}", __name__)
    estimators[estimator_name] = getattr(estimator_module, estimator_class)
