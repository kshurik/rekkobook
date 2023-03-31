import logging
from typing import Dict, List

import catboost as cb
import pandas as pd


class Ranker:
    def __init__(self, model_path: str = "./artefacts/catboost_clf.cbm"):
        logging.info("loading the model")
        self.ranker = cb.CatBoostClassifier().load_model(fname=model_path)

    @staticmethod
    def fit(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame = None,
        y_test: pd.DataFrame = None,
        ranker_params: dict = None,
        categorical_cols: list = None,
    ) -> None:
        """
        trains catboost clf model
        :X_train:
        :y_train:
        :X_test:
        :y_test:
        :ranker_params
        """

        logging.info(f"init ranker model with params {ranker_params}")
        cbm_classifier = cb.CatBoostClassifier(
            loss_function=ranker_params.get("loss_function", "CrossEntropy"),
            iterations=ranker_params.get("iterations", 5000),
            learning_rate=ranker_params.get("lr", 0.1),
            depth=ranker_params.get("depth", 6),
            random_state=ranker_params.get("random_state", 1234),
            verbose=ranker_params.get("verbose", True),
        )

        logging.info("started fitting the model")
        cbm_classifier.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=100,  # to avoid overfitting,
            cat_features=categorical_cols,
        )

        cbm_classifier.save_model(
            ranker_params.get("ranker_path", "./artefacts/catboost_clf.cbm")
        )

    def infer(self, ranker_input: List) -> Dict[str, int]:
        """
        inference for the output from lfm model
        :user_id:
        :candidates: dict with ranks {"item_id": 1, ...}
        """

        logging.info("making predictions...")
        preds = self.ranker.predict_proba(ranker_input)[:, 1]

        return preds
