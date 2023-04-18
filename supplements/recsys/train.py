# Here, you run both models training pipeline using modules we created
# LFM - load wathch data and run fit() method
# Ranker - load candidates based data with features and run fit() method
# REMINDER: it must be active and working. Before that, you shoul finalize prepare_ranker_data.py

import pandas as pd

from models.lfm import LFMModel
from models.ranker import Ranker
from utils.utils import read_parquet_from_gdrive

from fire import Fire

import logging

def train_lfm(data_path: str = None) -> None:
    """
    trains model for a given data with interactions
    :data_path: str, path to parquet with interactions
    """
    logging.INFO('Reading data...')
    if data_path is None:
        data = read_parquet_from_gdrive('https://drive.google.com/file/d/1MomVjEwY2tPJ845zuHeTPt1l53GX2UKd/view?usp=share_link')

    else:
        data = pd.read_parquet(data_path)

    logging.INFO('Started training LightFM model...')
    lfm = LFMModel()
    lfm.fit(data, user_col='user_id', item_col='item_id', model_params={})
    logging.INFO('Finished training LightFM model!')

def train_ranker():
    pass
