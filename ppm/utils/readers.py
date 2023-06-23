import pandas as pd
import zipfile

def reader(url: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(url,
                           delimiter = ';')
    except:
        data = pd.read_csv(url,
                           sep=';', 
                           encoding='latin-1')
    return data

def unzip_file(zip_path: str,
               extract_path: str) -> bool:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return True
