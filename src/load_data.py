import matplotlib
matplotlib.use('Agg') # [중요] 서버에서 GUI 창 끄기 (반드시 최상단)
import pandas as pd


def clean_and_parse_numeric(value):
    """
    문자열에서 괄호 '(', ')'를 제거하고 실수형(float)으로 변환하는 함수
    예: "(123.45)" -> 123.45
    """
    try:
        if isinstance(value, str):
            # 괄호 제거
            clean_str = value.replace('(', '').replace(')', '')
            return float(clean_str)
        return float(value)
    except (ValueError, TypeError):
        return None

def load_dat_file(file):
    """
    .dat 파일을 읽어와서 데이터프레임으로 반환하는 함수
    """
    df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, dtype=str)
    
    for col in df.columns:
        df[col] = df[col].apply(clean_and_parse_numeric)

    df = df.dropna()
    return df
