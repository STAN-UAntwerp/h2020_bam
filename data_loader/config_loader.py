import anyconfig
import pathlib
from pydantic import BaseModel, parse_obj_as
from logging_util.logger import get_logger

logger = get_logger(__name__)


class DataConfig(BaseModel):
    target: str
    test_size: float
    validation_size: float
    categorical_var: list[str]
    numerical_var: list[str]
    boolean_var: list[str]


def load_config() -> DataConfig:
    logger.debug("Loading data configuration")
    raw_config = anyconfig.load(pathlib.Path(__file__).parent.resolve() / 'config.yaml')
    return parse_obj_as(DataConfig, raw_config) 