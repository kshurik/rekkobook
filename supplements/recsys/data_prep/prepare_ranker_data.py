# TODO
# WRITE PIPELINE FOR DATA PREPARATION IN HERE TO USE FOR RANKER TRAININIG PIPELINE
from typing import Any, Dict, List
from utils.utils import read_parquet_from_gdrive
from configs.config import settings



def prepare_data_for_train(paths_config: Dict[str, str]):
    """
    function to prepare data to train catboost classifier.
    Basically, you have to wrap up code from full_recsys_pipeline.ipynb
    where we prepare data for classifier. In the end, it should work such
    that we trigger and use fit() method from ranker.py

        paths_config: dict, wher key is path name and value is the path to data
    """
    pass


def get_items_features(item_ids: List[int], item_cols: List[str]) -> Dict[int, Any]:
    """
    function to get items features from our available data
    that we used in training (for all candidates)
        :item_ids:  item ids to filter by
        :item_cols: feature cols we need for inference

    EXAMPLE OUTPUT
    {
    9169: {
    'content_type': 'film',
    'release_year': 2020,
    'for_kids': None,
    'age_rating': 16
        },

    10440: {
    'content_type': 'series',
    'release_year': 2021,
    'for_kids': None,
    'age_rating': 18
        }
    }

    """
    item_features = read_parquet_from_gdrive('https://drive.google.com/file/d/1XGLUhHpwr0NxU7T4vYNRyaqwSK5HU3N4/view?usp=share_link')
    item_features = item_features.set_index('item_id')
    item_features = item_features.to_dict('index')

    # collect all items
    output = {}
    for id in item_ids:
        output[id] = {k: v for k, v in item_features.get(id).items() if k in item_cols}

    return output


def get_user_features(user_id: int, user_cols: List[str]) -> Dict[str, Any]:
    """
    function to get user features from our available data
    that we used in training
        :user_id: user id to filter by
        :user_cols: feature cols we need for inference

    EXAMPLE OUTPUT
    {
        'age': None,
        'income': None,
        'sex': None,
        'kids_flg': None
    }
    """
    users = read_parquet_from_gdrive('https://drive.google.com/file/d/1MCTl6hlhFYer1BTwjzIBfdBZdDS_mK8e/view?usp=share_link')
    users = users.set_index('user_id')
    users_dict = users.to_dict('index')
    return {k: v for k, v in users_dict.get(user_id).items() if k in user_cols}

def prepare_ranker_input(
    candidates: Dict[int, int],
    item_features: Dict[int, Any],
    user_features: Dict[int, Any],
    ranker_features_order,
):
    ranker_input = []
    for k in item_features.keys():
        item_features[k].update(user_features)
        item_features[k]["rank"] = candidates[k]
        item_features[k] = {
            feature: item_features[k][feature] for feature in ranker_features_order
        }
        ranker_input.append(list(item_features[k].values()))

    return ranker_input
