import pandas as pd

from models.lfm import LFMModel
from models.ranker import Ranker
from utils.utils import read_parquet_from_gdrive
from data_prep.prepare_ranker_data import prepare_data_for_train

from fire import Fire

import logging

def train_lfm(data_path: str = None) -> None:
    """
    trains model for a given data with interactions
    :data_path: str, path to parquet with interactions
    """
    if data_path is None:
        logging.warning('Local data path is not set... Using default from GDrive')
        data = read_parquet_from_gdrive('https://drive.google.com/file/d/1MomVjEwY2tPJ845zuHeTPt1l53GX2UKd/view?usp=share_link')

    else:
        logging.info(f'Reading data from local path: {data_path}')
        data = pd.read_parquet(data_path)

    logging.info('Started training LightFM model...')
    lfm = LFMModel(is_infer = False) # train mode
    lfm.fit(
        data,
        user_col='user_id',
        item_col='item_id'
    )
    logging.info('Finished training LightFM model!')

def train_ranker():
    """
    executes training pipeline for 2nd level model
    all params are stored in configs
    """

    X_train, X_test, y_train, y_test = prepare_data_for_train()
    ranker = Ranker(is_infer = False) # train mode
    ranker.fit(X_train, y_train, X_test, y_test)
    logging.info('Finished training Ranker model!')

if __name__ == '__main__':
    Fire(
    {
        'train_lfm': train_lfm,
        'train_cbm': train_ranker
        }
    )