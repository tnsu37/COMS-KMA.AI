"""
[결측치 채우기 스크립트 - v8] (일별 처리 / 병렬 최적화 / 컬럼 불일치 오류 2종 모두 수정)

기능:
1. HOURLY_DATA 폴더를 스캔하여 처리할 *일(Day)* 목록을 찾습니다.
2. [v6] 날짜 범위(--start-date, --end-date) 옵션으로 병렬 처리를 지원합니다.
3. [v6] *하루(Day)* 단위로 데이터를 로드, Imputation, 저장합니다. (메모리 최적화)
4. [v8] (입력 오류 수정) Imputer가 훈련 시 본 컬럼(Feature 불일치) 오류 수정
5. [v8] (출력 오류 수정) Imputer가 전부 NaN인 컬럼(Shape 불일치) 오류 수정
6. [v6] --resume 기능이 일(Day) 단위로 작동합니다.

"""



import os
import pandas as pd
import glob
from collections import defaultdict
import argparse
import sys
from datetime import datetime, timedelta
import re
import warnings
import time
import joblib 
import numpy as np 
import gc
import logging
import traceback
#from missingpy import MissForest
from variable_mapping import get_daily_fill_columns

# pandas 경고 무시
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
DAILY_FILL_COLUMNS = get_daily_fill_columns()
# --- ProgressBar 클래스 ---
class ProgressBar:
    def __init__(self, total: int, desc: str = "", width: int = 30):
        self.total = total; self.current = 0; self.desc = desc; self.width = width
        self.start_time = time.time(); self.last_update = 0
    def update(self, n: int = 1):
        self.current += n; now = time.time()
        if now - self.last_update < 0.1 and self.current < self.total: return
        self.last_update = now; percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        filled = int(self.width * self.current / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (self.width - filled); elapsed = now - self.start_time
        eta_str = self._format_time((elapsed / self.current) * (self.total - self.current)) if self.current > 0 else "--:--"
        msg = f"\r{self.desc} |{bar}| {self.current}/{self.total} [{percentage:5.1f}%] [{self._format_time(elapsed)}<{eta_str}]"
        sys.stdout.write(msg); sys.stdout.flush()
        if self.current >= self.total: sys.stdout.write("\n"); sys.stdout.flush()
    def _format_time(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60); return f"{m:02d}:{s:02d}"
    def close(self):
        if self.current < self.total: self.current = self.total; self.update(0)
# ---------------------------

# +++ 로깅 설정 함수 +++
def setup_logger(log_file):
    logger = logging.getLogger('ImputerApply')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    error_console_handler = logging.StreamHandler(sys.stderr)
    error_console_handler.setLevel(logging.ERROR)
    error_console_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    error_console_handler.setFormatter(error_console_formatter)
    logger.addHandler(error_console_handler)
    
    return logger
# ++++++++++++++++++++++

logger = None # 전역 로거 변수

# 특정 케이스 고정값(평균)
FIXED_MEANS = {
    "ctps_cth": 4.179584097665568,
    "ctps_ctp": 614.7220684323216,
    "ctps_ctt": 254.53789325394524,
    "dcoew_radius": 20.571707302585317,
    "dcoew_thickness": 13.293504015554864,
    "dcoew_liquid_path": 94.42700552605407,
    "ncot": 5.019284345219368,
}

VALID_RANGE_GK2A = {
        'st_lon': {'type': 'con', 'values': [-np.inf, np.inf]},
        'st_lat': {'type': 'con', 'values': [-np.inf, np.inf]},
        'year': {'type': 'con', 'values': [2020, 2100]},
        'month': {'type': 'con', 'values': [1,12]},
        'day': {'type': 'con', 'values': [1,31]},
        'hour': {'type': 'con', 'values': [0,23]},
        'cld': {'type': 'cat', 'values': [0,1,2]},
        'ctps_cp': {'type': 'cat', 'values': [0,1,2,6]},
        'ctps_cth': {'type': 'con', 'values': [0,1700]},
        'ctps_ctp': {'type': 'con', 'values': [0,120000]},
        'ctps_ctt': {'type': 'con', 'values': [0,35000]},
        'cla_type': {'type': 'cat', 'values': [0,1,2,3,4,5,6,7,8,9]},
        'cla_cloud_fraction': {'type': 'con', 'values': [0,100]},
        'dcoew_radius': {'type': 'con', 'values': [2,90]},
        'dcoew_thickness': {'type': 'con', 'values': [1,160]},
        'dcoew_liquid_path': {'type': 'con', 'values': [25,1000]},
        'ncot': {'type': 'con', 'values': [-np.inf, np.inf]},
        #'ci_ci1': {'type': '', 'values': []},
        'ci_ci1_ccm': {'type': 'cat', 'values': [0,1,2,3,4,9]},
        #'ci_ci1_obj': {'type': '', 'values': []},
        #'ci_ci1_prob': {'type': 'cat', 'values': [0,1,2,3,4]},
        'ci_ci2': {'type': 'con', 'values': [-np.inf, np.inf]},
        #'ci_ci2_obj': {'type': '', 'values': []},
        #'fog': {'type': 'cat', 'values': [1,2,3,4,5,6,7]},
        'rr': {'type': 'con', 'values': [0,100]},
        'qpn_rate': {'type': 'con', 'values': [0,300]},
        #'qpn_probability': {'type': 'con', 'values': [0,100]},
        #'tqprof_q': {'type': 'con', 'values': [0,100]},
        #'tqprof_t': {'type': 'con', 'values': [180,320]},
        'tpw_low': {'type': 'con', 'values': [0, np.inf]},
        'tpw_mid': {'type': 'con', 'values': [0, np.inf]},
        'tpw_high': {'type': 'con', 'values': [0, np.inf]},
        'tpw': {'type': 'con', 'values': [0, 100]},
        #'aii_cape': {'type': 'con', 'values': [0,5000]},
        #'aii_ki': {'type': 'con', 'values': [0,40]},
        #'aii_li': {'type': '', 'values': []},
        #'aii_si': {'type': '', 'values': []},
        'aii_tti': {'type': 'con', 'values': [-43,56]},
        #'apps_aep': {'type': 'con', 'values': [-0.5,3]},
        #'apps_aod': {'type': 'con', 'values': [0,5]},
        #'apps_daod055': {'type': 'con', 'values': [0,5]},
        #'apps_daod11': {'type': 'con', 'values': [0,5]},
        #'lst': {'type': 'con', 'values': [213,330]},
        #'sal_bsa': {'type': 'con', 'values': [0,10000]},
        #'sal_bsa_b01': {'type': 'con', 'values': [0,10000]},
        #'sal_bsa_b02': {'type': 'con', 'values': [0,10000]},
        #'sal_bsa_b03': {'type': 'con', 'values': [0,10000]},
        #'sal_bsa_b04': {'type': 'con', 'values': [0,10000]},
        #'sal_bsa_b06': {'type': 'con', 'values': [0,10000]},
        #'sal_wsa': {'type': 'con', 'values': [0,10000]},
        #'sal_wsa_b01': {'type': 'con', 'values': [0,10000]},
        #'sal_wsa_b02': {'type': 'con', 'values': [0,10000]},
        #'sal_wsa_b03': {'type': 'con', 'values': [0,10000]},
        #'sal_wsa_b04': {'type': 'con', 'values': [0,10000]},
        #'sal_wsa_b06': {'type': 'con', 'values': [0,10000]},
        #'vgt_ndvi': {'type': 'con', 'values': [0,1]},
        #'vgt_evi': {'type': 'con', 'values': [0,1]},
        'swrad_downward': {'type': 'con', 'values': [0, np.inf]},
        'swrad_absorbed': {'type': 'con', 'values': [0, np.inf]},
        #'lwrad_downward': {'type': '', 'values': []},
        #'lwrad_upward': {'type': '', 'values': []},
        'ctps_dqf1': {'type': 'cat', 'values': [0,1,2,3,4,5,6,7,8]},
        'cla_type_dqf': {'type': 'cat', 'values': [0,1,2,3,4,5,6]},
        'cla_cloud_fraction_dqf': {'type': 'cat', 'values': [0,1,2,3,4,5]},
        'dcoew_dqf1': {'type': 'cat', 'values': [0,1,2,3,4,5,6,7,8]},
        'ncot_dqf': {'type': 'cat', 'values': [0,1,2,3,4,5,6]},
        'rr_raining_ct_flag': {'type': 'con', 'values': [1,20]},
        'qpn_dqf1': {'type': 'cat', 'values': [-1,0,1]},
        'tpw_dqf1': {'type': 'cat', 'values': [0,1,2]},
        'tpw_dqf2': {'type': 'cat', 'values': [0,1]},
        'aii_dqf1': {'type': 'cat', 'values': [0,1,2,3]},
        'aii_dqf2': {'type': 'cat', 'values': [0,1]},
        'swrad_downward_dqf': {'type': 'cat', 'values': [0,1]},
        'swrad_absorbed_dqf': {'type': 'cat', 'values': [0,1]},
        'swrad_dqf1': {'type': 'cat', 'values': [0,1]}
}

GK2A_CATEGORICAL_COLS = [
    col for col, info in VALID_RANGE_GK2A.items() if info["type"] == "cat"
]

GK2A_CONTINUOUS_COLS = [
    col for col, info in VALID_RANGE_GK2A.items() if info["type"] == "con"
]

# --- 헬퍼 함수 ---

SOURCE_DIR_MAP = {
    'GK2A': 'KOMPSAT_LE2',
    'ODAM': 'WEATHER_ODAM',
    'GEMS': 'GEMS'
}

def apply_categorical_outlier_to_99(df):
    for col, meta in VALID_RANGE_GK2A.items():
        if meta.get("type") == "cat" and col in df.columns:
            valid_values = set(meta["values"])
            mask = (~df[col].isin(valid_values)) & (~df[col].isna())
            df.loc[mask, col] = 99
    return df

def rule_based_imputation(df):
    """GK2A의 특정 변수의 특성을 반영하여 룰 기반 보간을 우선 수행합니다"""

    df = apply_categorical_outlier_to_99(df)

    for col in ["swrad_downward", "swrad_absorbed"]:
        if col in df.columns: df[col] = df[col].fillna(0)

    has_cols_1 = all(c in df.columns for c in ["cld", "ctps_cp"])
    has_cols_2 = all(c in df.columns for c in ["cld", "ctps_cp", "cla_type"])

    # 1. [ctps_cth, ctps_ctp, ctps_ctt] = if (cld=2 & ctps_cp=0)
    if has_cols_1:
        cond = (df["cld"] == 2) & (df["ctps_cp"] == 0)
        for col in ["ctps_cth", "ctps_ctp", "ctps_ctt"]:
            if col in df.columns:
                mask = cond & df[col].isna()
                df.loc[mask, col] = FIXED_MEANS[col]

    # 2. [dcoew_radius, docew_thickness] = if (cld=2 & ctps_cp=0) else if (cla_type=255)
    if has_cols_2:
        cond1 = (df["cld"] == 2) & (df["ctps_cp"] == 0)
        cond2 = (df["cla_type"] == 99)

        for col in ["dcoew_radius", "dcoew_thickness"]:
            if col in df.columns:
                mask1 = cond1 & df[col].isna()
                df.loc[mask1, col] = FIXED_MEANS[col]
                mask2 = (~cond1) & cond2 & df[col].isna()
                df.loc[mask2, col] = FIXED_MEANS[col]

    # 3. dcoew_liquid_path = if not (cld=0 & ctps_cp=1)
    if has_cols_1 and "dcoew_liquid_path" in df.columns:
        cond = ~((df["cld"] == 0) & (df["ctps_cp"] == 1))
        mask = cond & df["dcoew_liquid_path"].isna()
        df.loc[mask, "dcoew_liquid_path"] = FIXED_MEANS["dcoew_liquid_path"]

    # 4. ncot = if (cld=2)
    if "cld" in df.columns and "ncot" in df.columns:
        cond = (df["cld"] == 2)
        mask = cond & df["ncot"].isna()
        df.loc[mask, "ncot"] = FIXED_MEANS["ncot"]

    return df

def nearest_valid_category(x, valid_values):
    if pd.isna(x):
        return np.nan
    return min(valid_values, key=lambda v: abs(v - x))

def postprocess_imputed_df(df, data_source, valid_range):
    if df.empty:
        return df

    out = df.copy()

    if data_source != "GK2A":
        return out

    # 범주형 → nearest valid
    for col, info in valid_range.items():
        if col not in out.columns:
            continue

        if info["type"] == "cat":
            valid_values = info["values"]
            out[col] = out[col].apply(lambda x: nearest_valid_category(x, valid_values))

    return out

def calculate_pivot_date(data_date, first_pivot, pivot_interval_days):
    """데이터 날짜에 맞는 pivot 날짜 계산"""
    data_dt = datetime.strptime(data_date, "%Y-%m-%d")
    first_pivot_dt = datetime.strptime(first_pivot, "%Y-%m-%d")
    
    # 데이터 날짜가 first_pivot보다 이전이면 first_pivot 사용
    if data_dt < first_pivot_dt:
        return first_pivot
    
    # 데이터 날짜와 first_pivot 간의 차이(일)
    days_diff = (data_dt - first_pivot_dt).days
    
    # 몇 번째 interval에 속하는지 계산
    interval_index = days_diff // pivot_interval_days
    
    # 해당 interval의 pivot 날짜 계산
    pivot_dt = first_pivot_dt + timedelta(days=interval_index * pivot_interval_days)
    
    return pivot_dt.strftime('%Y-%m-%d')

def find_files_by_day_and_pivot(hourly_dir, data_source, first_pivot, pivot_interval_days, start_date=None, end_date=None):
    """상HOURLY_DATA 폴더를 스캔하여 일별 파일 맵 및 날짜-pivot 맵 반환 (날짜 범위 필터링)"""
    
    source_dir_name = SOURCE_DIR_MAP.get(data_source, data_source)
    scan_dir = os.path.join(hourly_dir, source_dir_name)
    
    logger.info(f"📂 스캔 중 (Pivot 기반, Interval: {pivot_interval_days}일): {scan_dir}")
    logger.info(f"  → First Pivot: {first_pivot}")
    
    day_file_map = defaultdict(list)
    day_to_pivot_map = {}
    
    search_path = os.path.join(scan_dir, "*", "*", "*", "*.csv")
    date_regex = re.compile(r"(\d{4}-\d{2}-\d{2})_\d{2}\.csv$")
    all_files = glob.glob(search_path)
    
    if not all_files:
        logger.error(f"❌ '{scan_dir}'에서 처리할 CSV 파일을 찾을 수 없습니다.")
        return None, None
    
    pbar = ProgressBar(total=len(all_files), desc="[파일 스캔]")
    for filepath in all_files:
        match = date_regex.search(os.path.basename(filepath))
        if match:
            date_str = match.group(1)
            
            if start_date and date_str < start_date:
                pbar.update(1); continue
            if end_date and date_str > end_date:
                pbar.update(1); continue
            
            day_file_map[date_str].append(filepath) 
            try:
                if date_str not in day_to_pivot_map: 
                    pivot_date = calculate_pivot_date(date_str, first_pivot, pivot_interval_days)
                    day_to_pivot_map[date_str] = pivot_date
            except ValueError: pass
        pbar.update(1)
    pbar.close()
    
    if not day_file_map:
        logger.error(f"❌ '{scan_dir}'에서 유효한 CSV 파일을 찾을 수 없습니다. (범위: {start_date}~{end_date})")
        return None, None
        
    logger.info(f"  → {len(day_file_map)}개 날짜 데이터 발견. (범위: {start_date or '처음'}~{end_date or '끝'})")
    return day_file_map, day_to_pivot_map

def load_daily_data(file_list, date_key):
    """특정 날짜(Day)의 모든 시간별 CSV를 읽어 하나의 DataFrame으로 합침"""
    logger.info(f"  → 讀 {date_key} 데이터 로드 중 ({len(file_list)}개 파일)...")
    df_list = []
    errors = 0
    for f in file_list:
        try: 
            df = pd.read_csv(f)
            df['dateTime'] = pd.to_datetime(df['dateTime'])
            df_list.append(df)
        except pd.errors.EmptyDataError: pass
        except Exception as e: 
            logger.warning(f"    - ⚠️ {os.path.basename(f)} 읽기 오류: {e}")
            errors += 1
    if errors > 0: logger.warning(f"    - {date_key} 로드 중 {errors}개 파일 오류 발생.")
    if not df_list: logger.error(f"    - {date_key} 유효 데이터 없음."); return None
    
    try:
        full_day_df = pd.concat(df_list, ignore_index=True)
        logger.info(f"    → ✓ 로드 및 병합 완료 (총 {len(full_day_df)}행)")
        return full_day_df
    except Exception as e:
        logger.exception(f"    - ❌ {date_key} 병합 실패 (메모리 부족 가능성): {e}")
        return None

def separate_data_for_imputation(df):
    """데이터 분리: (키 컬럼 DF, 데이터 컬럼 DF, 데이터 컬럼명 리스트) 반환"""
    logger.info(f"    → ⚙️ Imputation 적용 데이터 분리 중...")
    
    key_cols = ['geoId', 'geo_lon', 'geo_lat', 'dateTime']
    drop_cols = [
        'ci_ci1', 'ci_ci1_obj', 'ci_ci1_prob', 'ci_ci2_obj', 'fog',
        'qpn_probability', 'tqprof_q', 'tqprof_t', 'aii_cape', 'aii_ki',
        'aii_li', 'aii_si', 'apps_aep', 'apps_aod', 'apps_daod055', 'apps_daod11',
        'lst', 'sal_bsa', 'sal_bsa_b01', 'sal_bsa_b02', 'sal_bsa_b03', 'sal_bsa_b04',
        'sal_bsa_b06', 'sal_wsa', 'sal_wsa_b01', 'sal_wsa_b02', 'sal_wsa_b03',
        'sal_wsa_b04', 'sal_wsa_b06', 'vgt_ndvi', 'vgt_evi', 'lwrad_downward', 'lwrad_upward',
        'apps_aep_dqf', 'apps_aod_dqf', 'apps_daod055_dqf', 'apps_daod11_dqf', 'apps_daod011_dqf',
        'lst_dqf', 'lwrad_downward_dqf', 'lwrad_dqf1', 'lwrad_upward_dqf',
        'sal_dqf1', 'sal_dqf2', 'tqprof_dqf1', 'tqprof_dqf2', 'vgt_dqf1', 'fog_dqf', 'id' 
                       ]
    
    key_cols_existing = [col for col in key_cols if col in df.columns]
    key_df = df[key_cols_existing].copy()
    
    cols_to_drop = key_cols + drop_cols
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
    data_df = df.drop(cols_to_drop_existing, axis=1)
    logger.info(f"      - 제거된 컬럼: {len(cols_to_drop_existing)}개")
    
    for col in data_df.columns:
        if not pd.api.types.is_numeric_dtype(data_df[col]):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    non_numeric_cols = data_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        logger.info(f"      - 숫자형 아닌 컬럼 제외: {non_numeric_cols}")
        data_df = data_df.drop(non_numeric_cols, axis=1)
        
    data_col_names = data_df.columns.tolist()
    logger.info(f"      - 적용 대상 컬럼: {len(data_col_names)}개")
    return key_df, data_df, data_col_names

def save_imputed_hourly_files(imputed_df, imputed_dir, data_source):
    """Imputation 완료된 (일별) DF를 시간별 CSV로 분할하여 저장"""
    logger.info(f"    → 💾 Imputed 데이터 시간별 저장 중...")
    try:
        if 'dateTime' not in imputed_df.columns:
             logger.error("    - ❌ 'dateTime' 컬럼이 없어 저장할 수 없습니다."); return 0
        
        output_source_name = SOURCE_DIR_MAP.get(data_source, data_source)
        
        saved_count = 0
        imputed_df['kst_date_str'] = imputed_df['dateTime'].dt.strftime('%Y-%m-%d')
        imputed_df['kst_hour'] = imputed_df['dateTime'].dt.hour
        
        dt = imputed_df['dateTime'].iloc[0]
        date_str = dt.strftime('%Y-%m-%d')
        year, month, day = str(dt.year), str(dt.month), str(dt.day)

        pbar = ProgressBar(total=24, desc=f"    [저장: {date_str}]")

        for hour, hourly_df_group in imputed_df.groupby('kst_hour'):
            if hourly_df_group.empty: pbar.update(1); continue
            
            hourly_df_with_duplicates = hourly_df_group.copy()
            logger.debug(f"      [Hour {hour}] groupby 전 컬럼: {len(hourly_df_with_duplicates.columns)}개")
            hourly_df_with_duplicates.sort_values(by='dateTime', ascending=True, inplace=True)
            hourly_df = hourly_df_with_duplicates.groupby('geoId').last().reset_index()
            logger.debug(f"      [Hour {hour}] groupby 후 컬럼: {len(hourly_df.columns)}개")
            if hourly_df.empty: pbar.update(1); continue
            
            hour_str = f"{hour:02d}"
            output_path = os.path.join(imputed_dir, output_source_name, year, str(int(month)), str(day))
            os.makedirs(output_path, exist_ok=True)
            
            filename = f"{output_source_name}_{date_str}_{hour_str}.csv"
            filepath = os.path.join(output_path, filename)

            if 'geoId' in hourly_df.columns: hourly_df['geoId'] = hourly_df['geoId'].astype('Int64')
            if 'dateTime' in hourly_df.columns and pd.api.types.is_datetime64_any_dtype(hourly_df['dateTime']):
                 hourly_df['dateTime'] = hourly_df['dateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

            hourly_df.drop(['kst_date_str', 'kst_hour'], axis=1, inplace=True, errors='ignore')

            try:
                from variable_mapping import get_final_columns
                final_cols_no_flags = [c for c in get_final_columns() if '_dqf' not in c and '_flag' not in c]
                cols_ordered = [c for c in final_cols_no_flags if c in hourly_df.columns]
                
                # 교집합이 key 컬럼(4개)보다 많을 때만 재정렬 (실제 GK2A 데이터)
                # 테스트 데이터 등 다른 컬럼 이름을 사용할 경우 원본 유지
                if len(cols_ordered) > 4:
                    hourly_df = hourly_df[cols_ordered]
                    logger.debug(f"      [Hour {hour}] variable_mapping 적용: {len(cols_ordered)}개 컬럼")
                else:
                    logger.debug(f"      [Hour {hour}] variable_mapping 스킵 (교집합 {len(cols_ordered)}개만 발견)")
            except ImportError:
                pass 

            hourly_df.to_csv(filepath, index=False)
            saved_count += 1
            pbar.update(1)
            
        remaining_updates = 24 - (pbar.current % 24)
        if remaining_updates < 24: pbar.update(remaining_updates)
        pbar.close()

        logger.info(f"    → ✓ {saved_count}개 시간별 파일 저장 완료.")
        return saved_count

    except Exception as e:
        logger.exception(f"    - ❌ 시간별 저장 중 오류: {e}"); return 0

def check_if_day_processed(date_str, imputed_dir, data_source):
    """지정한 KST 날짜의 *폴더*가 존재하는지 확인"""
    try:
        source_dir_name = SOURCE_DIR_MAP.get(data_source, data_source)
        
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        year, month, day = str(dt.year), str(dt.month), str(dt.day)
        output_path = os.path.join(imputed_dir, source_dir_name, year, str(int(month)), str(day)) 
        return os.path.exists(output_path)
    except Exception:
        return False

# --- 메인 실행 ---
def main(args):
    """메인 실행 함수"""
    global logger
    
    hourly_dir = os.path.abspath(args.input)
    imputer_dir = os.path.abspath(args.input_imputer)
    imputed_dir = os.path.abspath(args.output)
    
    data_source = args.data_source
    if data_source == 'AIRKOREA':
        print("AIRKOREA는 Imputation 제외 대상입니다. 작업을 종료합니다.")
        sys.exit(0)
    
    first_pivot = args.first_pivot
    pivot_interval = args.pivot_interval
    resume = args.resume
    force = args.force
    start_date = args.start_date
    end_date = args.end_date

    log_dir = os.path.join(imputed_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_suffix = f"{data_source}_{(start_date or 'start').replace('-', '')}_to_{(end_date or 'end').replace('-', '')}"
    log_file = os.path.join(log_dir, f"impute_data_{log_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger(log_file)

    logger.info("="*70)
    logger.info(f"🛰️  [결측치 채우기] Pivot 기반 Imputation 적용 시작 (v10 - Pivot 모드)")
    logger.info(f"  - 데이터 소스: {data_source}")
    logger.info(f"  - 입력 경로 (HOURLY_DATA): {hourly_dir}")
    logger.info(f"  - Imputer 경로: {imputer_dir}")
    logger.info(f"  - 출력 경로 (IMPUTED_DATA): {imputed_dir}")
    logger.info(f"  - First Pivot: {first_pivot}")
    logger.info(f"  - Pivot Interval: {pivot_interval}일")
    logger.info(f"  - [v10] 처리 범위: {start_date or '처음'} 부터 {end_date or '끝'} 까지")
    if resume: logger.info("  - 모드: 🔄 Resume")
    if force: logger.info("  - 모드: ⚠️ Force")
    logger.info(f"  - 로그 파일: {log_file}")
    logger.info("="*70)

    try:
        day_file_map, day_to_pivot_map = find_files_by_day_and_pivot(hourly_dir, data_source, first_pivot, pivot_interval, start_date, end_date)
        if not day_file_map: return

        main_pbar = ProgressBar(total=len(day_file_map), desc="[전체 진행 (일별 Impute)]")
        total_files_saved = 0
        skipped_count = 0
        
        current_imputer = None
        current_pivot_date = None
        #expected_cols = None # [v8] Imputer가 기대하는 컬럼 목록

        for date_key in sorted(day_file_map.keys()):
            file_list = day_file_map[date_key]
            
            if resume and not force and check_if_day_processed(date_key, imputed_dir, data_source):
                logger.info(f"--- {date_key} ⏭️ 건너뛰기 (출력 폴더가 이미 존재함) ---")
                main_pbar.update(1); skipped_count += 1; continue

            logger.info(f"--- {date_key} Imputation 적용 시작 ---")
            
            try: pivot_date = day_to_pivot_map[date_key]
            except KeyError:
                logger.error(f"    - ❌ {date_key}에 해당하는 Pivot 날짜를 찾을 수 없음. 건너뜁니다.")
                main_pbar.update(1); continue

            if current_pivot_date != pivot_date or current_imputer is None:
                imputer_filename = f"{data_source.lower()}_imputer_{pivot_date}.pkl"
                imputer_filepath = os.path.join(imputer_dir, imputer_filename)
                logger.info(f"    → 讀 Imputer 로드 중: {imputer_filename} (Pivot: {pivot_date})")
                try:
                     current_imputer = joblib.load(imputer_filepath)
                     current_pivot_date = pivot_date
                     valid_range = VALID_RANGE_GK2A if data_source == "GK2A" else {}

                     # missingpy transform에서 NaN을 못 받는 문제 우회
                     current_imputer.missing_values = -9999
                     
                     logger.info(f"    → ✓ Pivot {pivot_date} Imputer 로드 완료.")

                except FileNotFoundError:
                     logger.error(f"    - ❌ Imputer 파일을 찾을 수 없습니다: {imputer_filepath}. 이 날짜를 건너뜁니다.")
                     main_pbar.update(1); continue
                except Exception as e:
                     logger.error(f"    - ❌ Imputer 로드 실패: {e}. 이 날짜를 건너뜁니다.")
                     main_pbar.update(1); continue
            
            day_df = load_daily_data(file_list, date_key)
            if day_df is None:
                logger.error(f"  - ❌ {date_key} 데이터 로드 실패. 건너뜁니다.")
                main_pbar.update(1); continue
            
            if data_source == "GK2A":
                day_df = rule_based_imputation(day_df)

            # --- [v9.2] SAL/VGT 00시 값 보간 (Imputation 전에 선행) ---
            # try:
                
            #     # GK2A 데이터에 대해서만 이 로직을 적용
            #     if data_source == 'GK2A' and DAILY_FILL_COLUMNS:
            #         existing_fill_cols = [col for col in DAILY_FILL_COLUMNS if col in day_df.columns]
            #         if existing_fill_cols:
            #             logger.info(f"    → 🔄 {date_key} SAL/VGT 00시 값 보간 적용 중...")
            #             day_df.sort_values(by=['geoId', 'dateTime'], inplace=True)
            #             # geoId별로 00시 값을 ffill/bfill (P1&2 로직과 동일)
            #             day_df[existing_fill_cols] = day_df.groupby('geoId')[existing_fill_cols].transform('ffill')
            #             day_df[existing_fill_cols] = day_df.groupby('geoId')[existing_fill_cols].transform('bfill')  
            # except Exception as e:
            #     logger.error(f"    - ❌ {date_key} SAL/VGT 보간 중 오류: {e}")
            # --------------------------------------------------------

            # [수정] data_col_names: Imputer 적용 전 원본 데이터의 컬럼 목록
            key_df, data_df, data_col_names = separate_data_for_imputation(day_df)
            original_index = data_df.index
            expected_cols = data_col_names.copy()
            del day_df
            gc.collect()

            if data_df.empty:
                logger.warning(f"  - ⚠️ {date_key} Imputation 적용 대상 데이터 없음. 빈 파일 생성.")
                imputed_df_data_part = pd.DataFrame(columns=data_col_names, index=original_index)
            else:
                logger.info(f"    → ✨ Imputation 적용 중 (transform)...")
                start_transform = time.time()
                try:
                    # --- [v9 "Shape" 및 "Feature" 오류 2종 동시 수정] ---
                    
                    # 1. Imputer가 훈련(fit) 시 기억하는 컬럼(55개)으로 현재 데이터를 재정렬
                    # -> "Feature names... missing" 오류 해결
                    
                    # 2. Imputer 실행 (set_output="pandas"로 인해 DataFrame 반환)
                    # (입력은 55개, Imputer가 처리한 53개 컬럼만 가진 DF가 반환됨)

                    data_for_transform = data_df.reindex(columns=expected_cols, index=original_index)

                    # missingpy transform 우회용 sentinel
                    data_for_transform = data_for_transform.fillna(-9999)

                    imputed_array = current_imputer.transform(data_for_transform.values)

                    imputed_df_processed = pd.DataFrame(
                        imputed_array,
                        columns=expected_cols,
                        index=original_index
                    )

                    transform_elapsed = time.time() - start_transform
                    
                    # 3. 원본 컬럼(data_col_names) 기준으로 최종 복원
                    # (imputed_df_processed(53개)를 data_col_names(55개) 기준으로 재정렬)
                    # -> "all-NaN" 2개 컬럼이 NaN으로 채워져 복원됨
                    imputed_df_data_part = imputed_df_processed.reindex(columns=data_col_names, index=original_index)

                    logger.info(f"    → ✓ 적용 완료 ({transform_elapsed:.1f}초)")
                    # --- [v9 오류 수정 완료] ---
                except Exception as e:
                    logger.exception(f"    - ❌ {date_key} Imputation 적용 실패: {e}")
                    main_pbar.update(1); continue
                finally:
                    # --- [v9.1 수정] ---
                    # 'imputed_data_array' 변수는 v9에 존재하지 않으므로 삭제
                    # 'data_df'는 항상 존재하므로 삭제
                    del data_df
                    # 'imputed_df_processed'는 except 발생 시 없을 수 있으므로 
                    # locals()로 확인 후 삭제 (안전하게)
                    if 'imputed_df_processed' in locals():
                        del imputed_df_processed
                    gc.collect()
                    # --------------------

            logger.info(f"    → 🧩 데이터 재조립 중...")
            logger.info(f"      - key_df 컬럼: {len(key_df.columns)}개 ({list(key_df.columns)})")
            logger.info(f"      - imputed_df_data_part 컬럼: {len(imputed_df_data_part.columns)}개")
            final_imputed_df = pd.concat([key_df, imputed_df_data_part], axis=1)
            final_imputed_df = postprocess_imputed_df(final_imputed_df, data_source, valid_range)
            logger.info(f"      - final_imputed_df 컬럼: {len(final_imputed_df.columns)}개")
            del key_df, imputed_df_data_part
            gc.collect()
            logger.info(f"    → ✓ 재조립 완료.")

            saved_count = save_imputed_hourly_files(final_imputed_df, imputed_dir, data_source)
            total_files_saved += saved_count
            del final_imputed_df
            gc.collect()

            main_pbar.update(1)

        main_pbar.close()
        logger.info("="*70)
        logger.info("🎉 [결측치 채우기] 작업이 완료되었습니다.")
        logger.info(f"  - 총 {total_files_saved}개의 시간별 파일이 생성/업데이트되었습니다.")
        if resume: logger.info(f"  - 총 {skipped_count}개 날짜를 건너뛰었습니다.")
        logger.info(f"최종 Imputed 데이터는 {imputed_dir} 에서 확인하세요.")
        logger.info(f"상세 로그는 {log_file} 에서 확인하세요.")
        logger.info("="*70)

    except Exception as e:
        logger.exception("💥 스크립트 실행 중 치명적인 오류 발생")
        print(f"❌ 치명적 오류 발생. {log_file} 파일을 확인하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="[결측치 채우기 v9] 일(Day) 단위로 Imputation을 적용합니다.")
    
    parser.add_argument( '--input', '-i', type=str, required=True, help="[입력] 시간별 CSV 파일이 있는 HOURLY_DATA 디렉토리 (예: /app/data/HOURLY_DATA)")
    parser.add_argument( '--input_imputer', '-imp', type=str, required=True, help="[입력] 기간별 Imputer(.pkl) 파일이 있는 디렉토리 (예: /app/data/imputers)")
    parser.add_argument( '--output', '-o', type=str, required=True, help="[출력] Imputation이 완료된 시간별 CSV 파일을 저장할 디렉토리 (예: /app/data/HOURLY_DATA_IMPUTED)")
    parser.add_argument( '--data-source', '-s', type=str, required=True, choices=['GK2A', 'ODAM', 'GEMS', 'AIRKOREA'], help='Imputation을 적용할 데이터 소스 (AIRKOREA는 제외)')
    parser.add_argument( '--first-pivot', '-fp', type=str, required=True,
                         help="첫 번째 pivot 날짜 (YYYY-MM-DD). 예: 2024-10-01")
    parser.add_argument( '--pivot-interval', '-pi', type=int, default=7,
                         help="Pivot 간격 (일). 기본값: 7일 (매주)")
    parser.add_argument( '--resume', action='store_true', help='중단된 작업 재개 (이미 처리된 날짜 건너뛰기)')
    parser.add_argument( '--force', action='store_true', help='기존 파일 덮어쓰기 (--resume 옵션 무시)')
    parser.add_argument( '--start-date', type=str, default=None, help="[v9] 처리를 시작할 KST 날짜 (YYYY-MM-DD)")
    parser.add_argument( '--end-date', type=str, default=None, help="[v9] 처리를 종료할 KST 날짜 (YYYY-MM-DD)")
    
    args = parser.parse_args()
    main(args)