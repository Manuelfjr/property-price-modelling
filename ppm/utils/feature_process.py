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