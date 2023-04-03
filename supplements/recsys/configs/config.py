from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=[
        "configs/user_features.toml",
    ]
)