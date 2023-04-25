from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=[
        "configs/features.toml",
        "configs/ranker_data_prep.toml",
        "configs/models_params.toml"
    ]
)