import json
from datetime import datetime, timedelta

import dill
import numpy as np
import pandas as pd
from cachetools import TTLCache, cached


@cached(cache=TTLCache(maxsize=1024, ttl=timedelta(hours=12), timer=datetime.now))
def read_parquet_from_gdrive(url):
    """
    gets csv data from a given url (from file -> share -> copy link)
    :url: *****/view?usp=share_link
    """
    file_id = url.split("/")[-2]
    file_path = "https://drive.google.com/uc?export=download&id=" + file_id
    data = pd.read_parquet(file_path)

    return data


def generate_lightfm_recs_mapper(
    model: object,
    item_ids: list,
    known_items: dict,
    user_features: list,
    item_features: list,
    N: int,
    user_mapping: dict,
    item_inv_mapping: dict,
    num_threads: int = 4,
):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.predict(
            user_id,
            item_ids,
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
        )

        additional_N = len(known_items[user_id]) if user_id in known_items else 0
        total_N = N + additional_N
        top_cols = np.argpartition(recs, -np.arange(total_N))[-total_N:][::-1]

        final_recs = [item_inv_mapping[item] for item in top_cols]
        if additional_N > 0:
            filter_items = known_items[user_id]
            final_recs = [item for item in final_recs if item not in filter_items]
        return final_recs[:N]

    return _recs_mapper


def save_model(model: object, path: str):
    with open(f"{path}", "wb") as obj_path:
        dill.dump(model, obj_path)


def load_model(path: str):
    with open(path, "rb") as obj_file:
        obj = dill.load(obj_file)
    return obj


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)
