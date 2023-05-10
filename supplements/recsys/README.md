# Full RecSys Pipeline
Here we have the full pipeline to train and make inference using two-level model architecture

## Repo Structure
- /artefacts - local storage for models artefacts;
- /configs - configuration files with [dynaconf](https://www.dynaconf.com/) for convenice development;
- /data_prep - data preparation modules to be used during training_pipeline;
- /models - model fit and inference pipeline;
- /utils - some common functions thatn can be used everywhere

## Commands
- `poetry install` - to install all dependencies. After that, additionally run `poetry run pip install lightfm==1.17 --no-use-pep517` to workaround install lfm model issue;
- `poetry run python train.py train_lfm` - runs training pipeline for candidates model (run within created env);
- `poetry run python train.py train_cbm` - runs ranker training pipeline (takes a while)
- `poetry run python api.py` - runs API locally (paste it in your browser to check http://localhost:8000/get_recommendation?user_id=176549&top_k=10 or use `request` module with user_id & top_k params to send request with base url http://127.0.0.1:8000/get_recommendation)

For any suggestions / issues dm @kshurik