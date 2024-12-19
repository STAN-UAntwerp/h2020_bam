import pandas as pd

from logging_util.logger import get_logger
logger = get_logger(__name__)


def load_data(target: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = pd.read_csv(f'../../data_{target}.csv', index_col=0)
    df_blanks = pd.read_csv('../../blanks.csv', index_col=0)
    
    return df, df_blanks
