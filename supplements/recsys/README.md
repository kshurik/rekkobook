# Full RecSys Pipeline
Here we have the full pipeline to train and make inference using two-level model architecture

## Repo Structure
- /artefacts - local storage for models artefacts;
- /data_prep - data preparation modules to be used during training_pipeline;
- /models - model fit and inference pipeline
- /utils - some common functions thatn can be used everywhere

## Commands
- `poetry install` - to install all dependencies. After that, additionally run `poetry run pip install lightfm==1.17 --no-use-pep517` to workaround install lfm model issue;
- `python train.py train_lfm` - runs training pipeline for candidates model (run within created env);
- `python train.py train_cbm` - runs ranker training pipeline (takes a while)


For any suggestions / issues dm @kshurik