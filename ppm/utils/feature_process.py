from pandas import DataFrame, concat
from tqdm import tqdm
def replace_names(value: str) -> str:
    value = value.replace('parking ','')
    value = value.replace('bedroom ','')
    value = value.replace('bathroom ','')
    value = value.replace('area ','')
    value = value.replace('m²','')
    value = value.replace(' ','')
    return value

def transform_float(value: str) -> float:
    value = value.replace("R$ ","")
    value = value.replace(".","")
    try:
        value = float(value)
    except:
        value = value
    return value

def convert_to_float(x: str) -> float:
    try:
        return float(x)
    except:
        return -1
    
def replace_na_with_mean(
        data_merged: DataFrame,
        columns_na: list,
        bairro: str) -> tuple:
    data_bairros = {}
    errors = []
    
    for bairro in tqdm(data_merged.bairro.unique()):
        data_bairros[bairro] = data_merged[data_merged.bairro.str.contains(bairro)].reset_index(drop=True)
        
        for na_col in tqdm(columns_na):
            try:
                data_bairros[bairro][na_col] = data_bairros[bairro][na_col].astype(float)
                average = data_bairros[bairro][na_col].mean()
                data_bairros[bairro][na_col] = data_bairros[bairro][na_col].fillna(average)
            except Exception as e:
                errors.append(e)
                continue
    
    return data_bairros, errors

def replace_na_with_mode(
        data_merged: DataFrame,
        columns_na: list,
        bairro: str) -> tuple:
    data_bairros = {}
    errors = []
    
    for bairro in tqdm(data_merged.bairro.unique()):
        data_bairros[bairro] = data_merged[data_merged.bairro.str.contains(bairro)].reset_index(drop=True)
        
        for na_col in tqdm(columns_na):
            try:
                if data_bairros[bairro][na_col].dtype == 'object':
                    mode_value = data_bairros[bairro][na_col].mode().iloc[0]
                    data_bairros[bairro][na_col] = data_bairros[bairro][na_col].fillna(mode_value)
                else:
                    data_bairros[bairro][na_col] = data_bairros[bairro][na_col].astype(float)
                    average = data_bairros[bairro][na_col].mean()
                    data_bairros[bairro][na_col] = data_bairros[bairro][na_col].fillna(average)
                    
            except Exception as e:
                errors.append(e)
                continue
    content = concat(list(data_bairros.values()), ignore_index = True)
    content[na_col] = content[na_col].fillna(content[na_col].mode())
    return content, errors