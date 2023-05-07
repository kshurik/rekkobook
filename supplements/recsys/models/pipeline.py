import logging

from configs.config import settings
from data_prep.prepare_ranker_data import (
    get_items_features,
    get_user_features,
    prepare_ranker_input,
)


def get_recommendations(
    user_id: int, lfm_model: object, ranker: object, top_k: int = 20
):
    """
    function to get recommendation for a given user id
    """

    try:
        logging.info("getting 1st level candidates")
        candidates = lfm_model.infer(user_id=user_id, top_k=top_k)

        logging.info("getting features...")
        user_features = get_user_features(user_id, user_cols=settings.USER_FEATURES)
        item_features = get_items_features(
            item_ids=list(candidates.keys()), item_cols=settings.ITEM_FEATURES
        )

        ranker_input = prepare_ranker_input(
            candidates=candidates,
            item_features=item_features,
            user_features=user_features,
            ranker_features_order=ranker.ranker.feature_names_,
        )
        preds = ranker.infer(ranker_input=ranker_input)
        predictions = dict(zip(candidates.keys(), preds))
        sorted_recommendations = dict(
            sorted(predictions.items(), key=lambda item: item[1], reverse=True)
        )

        output = {
            "recommendations": list(sorted_recommendations.keys()),
            "status": "success",
            "msg": None,
        }

    except Exception as e:
        output = {"recommendations": None, "status": "error", "msg": str(e)}

    return output
