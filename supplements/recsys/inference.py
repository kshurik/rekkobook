
from models.lfm import LFMModel
from models.ranker import Ranker

from configs.config import settings

from data_prep.prepare_ranker_data import (
    get_user_features,
    get_items_features,
    prepare_ranker_input
    )

import logging


def get_recommendations(user_id: int, top_k: int = 20):
    """
    function to get recommendation for a given user id
    """

    lfm_model = LFMModel()
    ranker = Ranker()

    logging.info('getting 1st level candidates')
    candidates = lfm_model.infer(user_id = user_id, top_k = top_k)

    logging.info('getting features...')
    user_features = get_user_features(user_id, user_cols=settings.USER_FEATURES)
    item_features = get_items_features(item_ids = list(candidates.keys()), item_cols = settings.ITEM_FEATURES)

    ranker_input = prepare_ranker_input(
        candidates = candidates,
        item_features = item_features,
        user_features=user_features,
        ranker_features_order=ranker.ranker.feature_names_
        )
    preds = ranker.infer(ranker_input = ranker_input)
    output = dict(zip(candidates.keys(), preds))

    return output

# if __name__ == '__main__':
#     print(get_recommendations(973171))
