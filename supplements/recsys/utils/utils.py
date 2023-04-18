import pandas as pd


def read_parquet_from_gdrive(url):
    """
    gets csv data from a given url (from file -> share -> copy link)
    :url: *****/view?usp=share_link
    """
    file_id = url.split("/")[-2]
    file_path = "https://drive.google.com/uc?export=download&id=" + file_id
    data = pd.read_parquet(file_path)

    return data
