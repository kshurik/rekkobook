import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset

from configs.config import settings
from utils.utils import load_model, save_model


class LFMModel:
    def __init__(self, is_infer=True):
        if is_infer:
            logging.info("loading candidates model")
            self.lfm_model = load_model(settings.LFM_TRAIN_PARAMS.MODEL_PATH)
            self.mapper = load_model(settings.LFM_TRAIN_PARAMS.MAPPER_PATH).mapping()

        else:
            pass

    @staticmethod
    def df_to_tuple_iterator(data: pd.DataFrame):
        """
        special iterator to use with lightfm
        :df: pd.DataFrame, interactions dataframe
        returs iterator
        """
        return zip(*data.values.T)

    def fit(self, data: pd.DataFrame, user_col: str, item_col: str) -> None:
        """
        Trains and saves model with mapper for further inference
        """

        # init class and fit tuple of user and movie interactions
        dataset = Dataset()
        dataset.fit(data[user_col].unique(), data[item_col].unique())

        # defining train set on the whole interactions dataset
        logging.info("Creating training matrix")
        train_mat, train_mat_weights = dataset.build_interactions(
            self.df_to_tuple_iterator(data[[user_col, item_col]])
        )

        # save mappers
        save_model(dataset, settings.LFM_TRAIN_PARAMS.MAPPER_PATH)

        # init model
        epochs = settings.LFM_TRAIN_PARAMS.EPOCHS
        lfm_model = LightFM(
            no_components=settings.LFM_TRAIN_PARAMS.NO_COMPONENTS,
            learning_rate=settings.LFM_TRAIN_PARAMS.LEARNING_RATE,
            loss=settings.LFM_TRAIN_PARAMS.LOSS,
            max_sampled=settings.LFM_TRAIN_PARAMS.MAX_SAMPLED,
            random_state=settings.LFM_TRAIN_PARAMS.RANDOM_STATE,
        )

        # execute training
        for i in range(epochs):
            logging.info(f"Epoch num {i} in LFM model training")
            lfm_model.fit_partial(train_mat, num_threads=4)

        # save model
        save_model(lfm_model, settings.LFM_TRAIN_PARAMS.MODEL_PATH)

    def infer(self, user_id: int, top_k: int = 20) -> Dict[str, int]:
        """
        method to make recommendations for a single user id
        :user_id: str, user id
        :model_path: str, relative path for the model
        """

        # set params
        user_row_id = self.mapper[0][user_id]
        all_items_list = list(self.mapper[2].values())

        preds = self.lfm_model.predict(user_row_id, all_items_list)

        # make final predictions
        item_inv_mapper = {v: k for k, v in self.mapper[2].items()}
        top_preds = np.argpartition(preds, -np.arange(top_k))[-top_k:][::-1]
        item_pred_ids = []
        for item in top_preds:
            item_pred_ids.append(item_inv_mapper[item])
        final_preds = {v: k + 1 for k, v in enumerate(item_pred_ids)}

        return final_preds
