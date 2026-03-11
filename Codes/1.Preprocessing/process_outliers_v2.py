"""
[프로세스 2: 이상치 처리 (통합 패키지)]

기능:
- --data-type 옵션에 따라 "과거 데이터(.nc)"와 "API 데이터(.csv)"를 모두 처리합니다.

[모드 1: GK2A_PAST]
- 'process_direct_hourly.py'의 로직을 "그대로" 수행합니다.
- 입력: 'sample' 폴더(.nc), 'matched_geoid.csv'
- 처리: 
  1. `variable_mapping.py`를 import 합니다. (중요)
  2. `matched_geoid.csv`를 읽어서(로드) 좌표 매칭 정보를 사용합니다.
  3. .nc 파일을 읽고 `index`로 데이터 추출, TQPROF(z=51) 처리
  4. `clean_outliers_past` (원본 컬럼명 기준) 이상치 처리
  5. `apply_variable_mapping` (이름 변경) 수행
  6. `KST` 변환 수행
  7. `SAL/VGT` 보간 (`ffill` + `bfill`) 수행
  8. `geoId` 중복 제거 후 시간별 CSV로 `output_dir`에 저장
- 출력: HOURLY_DATA (최종 88컬럼, DQF 포함)

[모드 2: GK2A_API, AIRKOREA_API, ODAM_API]
- 입력: API CSV 파일 1개
- 처리: 
  1. "최종 API 컬럼명" 기준으로 이상치 처리를 수행합니다.
  2. (variable_mapping.py import 불필요)
- 출력: 이상치 처리된 CSV 파일 1개
"""

import os
import sys
import pandas as pd
#import xarray as xr
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
import numpy as np
import logging
import json
import traceback
import time
import argparse
import psutil # 메모리 모니터링용
import gc
import re
from collections import defaultdict
import warnings # [수정] warnings import 추가
import glob
from pathlib import Path

# pandas 경고 무시
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- [수정] variable_mapping.py & process_all_sample.py 함수 임포트 ---
# (process_direct_hourly.py의 import 로직 "그대로" 사용)
try:
    from variable_mapping import (
        get_final_columns,
        get_daily_fill_columns,
        apply_variable_mapping # 이름 매핑용
    )
    MAPPING_ENABLED = True
    FINAL_COLUMNS = get_final_columns()
    DAILY_FILL_COLUMNS = get_daily_fill_columns()
    print("✅ 'variable_mapping.py'에서 모든 함수 로드 완료.")
except ImportError as e:
    print(f"⚠️ 'variable_mapping.py'를 찾을 수 없습니다: {e}")
    MAPPING_ENABLED = False
# --------------------------------------------

# --- [수정] 날짜 범위 설정 (GK2A_PAST 모드용) ---
# ⚠️ UTC 기준! KST 2023-01-01 00:00부터 처리하려면 UTC 2022-12-31 15:00부터 필요
DATA_START_DATE = datetime(2022, 12, 31, 15, 0, 0)  # UTC 2022-12-31 15:00 = KST 2023-01-01 00:00
DATA_END_DATE = datetime(2025, 5, 1, 8, 59, 59)  # UTC 2025-05-01 08:59 = KST 2025-05-01 17:59 (하루 마지막)
# --------------------------------------------

# +++ 로깅 설정 함수 +++
def setup_logger(log_file=None):
    """로거 설정 (파일 또는 콘솔)"""
    logger = logging.getLogger('ProcessOutliers')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 콘솔 핸들러 (기본)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (log_file 인자가 주어지면)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger
# ++++++++++++++++++++++

logger = setup_logger() # 전역 로거 (일단 콘솔만)

# --- ProgressBar 클래스 ---
class ProgressBar:
    def __init__(self, total: int, desc: str = "", width: int = 30):
        self.total = total; self.current = 0; self.desc = desc; self.width = width;
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
# -----------------

# ==============================================================================
# [모드 1] GK2A_PAST (.nc) 처리 로직
# (process_direct_hourly.py의 함수들 "그대로" 가져옴)
# ==============================================================================

# --- 1. GK2A_PAST: clean_outliers_past (원본.nc 컬럼명 기준) ---
# --- process_all_sample.py의 clean_outliers 함수 복사 ---
def clean_outliers_past(df, folder_name):
    # ... (이전과 동일) ...
    import numpy as np
    invalid_values = [99, -99, 999, -999, 65535]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(invalid_values, np.nan)
    if 'quality_flag1' in df.columns and 'quality_flag2' in df.columns:
        invalid_mask = (df['quality_flag1'] != 0) | (df['quality_flag2'] != 1)
        data_cols = [c for c in df.columns if c not in
                     ['geoId','geo_lon', 'geo_lat', 'st_lon', 'st_lat',
                      'index', 'matched_distance', 'dateTime']
                     and not c.startswith('quality_flag')
                     and not c.endswith('_dqf') and not c.endswith('_flag')]
        if data_cols:
            df.loc[invalid_mask, data_cols] = np.nan
    folder_upper = folder_name.upper()
    if folder_upper == 'SWRAD':
        # for col_orig in ['SW_DQF','DSR', 'ASR', 'RSR']:
            #  if col_orig in df.columns:
            #     dqf_col_orig = f"{col_orig}_DQF1"
            #     if dqf_col_orig in df.columns: df.loc[df[dqf_col_orig] != 1, col_orig] = np.nan
        for col_orig in ['SW_DQF']:
            if col_orig in df.columns:
                df.loc[df[col_orig] != 1, col_orig] = np.nan
    elif folder_upper == 'SAL':
        if 'DQF_BSA' in df.columns:
            bsa_cols = [c for c in df.columns if c.startswith('BSA')]; df.loc[df['DQF_BSA'] != 1, bsa_cols] = np.nan
        if 'DQF_WSA' in df.columns:
            wsa_cols = [c for c in df.columns if c.startswith('WSA')]; df.loc[df['DQF_WSA'] != 1, wsa_cols] = np.nan
    elif folder_upper == 'LWRAD':
        for col_orig in ['BEMIS', 'DLR', 'ULR', 'OLR', 'LWRAD_DQF']:
            if col_orig in df.columns:
                dqf_col_orig = f"{col_orig}_DQF"
                if dqf_col_orig in df.columns: df.loc[df[dqf_col_orig] != 1, col_orig] = np.nan
    elif folder_upper == 'NCOT':
        if 'NCOT_DQF' in df.columns and 'NCOT' in df.columns: df.loc[df['NCOT_DQF'] != 0, 'NCOT'] = np.nan
    elif folder_upper == 'LST':
        if 'DQF_LST' in df.columns and 'LST' in df.columns: df.loc[df['DQF_LST'] != 0, 'LST'] = np.nan
    elif folder_upper == 'FOG':
        if 'DQF_FOG' in df.columns: df.loc[df['DQF_FOG'] != 0, 'FOG'] = np.nan
            # fog_cols = [c for c in df.columns if c.startswith('FOG') and c != 'DQF_FOG']
            # df.loc[df['DQF_FOG'] != 0, fog_cols] = np.nan
    elif folder_upper == 'DCOEW':
        if 'DCOEW_DQF' in df.columns:
            dcoew_cols = [c for c in df.columns if c.startswith(('CER', 'COT', 'LWP'))]
            df.loc[df['DCOEW_DQF'] != 0, dcoew_cols] = np.nan
    elif folder_upper == 'CTPS':
        if 'CTPS_flag' in df.columns:
            # CTPS 제품의 모든 변수: CP, CTH, CTP, CTT, CLD_EMIS_11 및 각각의 _flag
            ctps_cols = [c for c in df.columns if c.startswith(('CP', 'CTH', 'CTP', 'CTT', 'CLD_EMIS'))]
            df.loc[df['CTPS_flag'] != 0, ctps_cols] = np.nan
    elif folder_upper == 'CLA':
        if 'CF_DQF' in df.columns and 'CF' in df.columns: df.loc[df['CF_DQF'] != 0, 'CF'] = np.nan
        if 'CT_DQF' in df.columns and 'CT' in df.columns: df.loc[df['CT_DQF'] != 0, 'CT'] = np.nan
    elif folder_upper == 'APPS':
        for col_orig in ['AEP', 'AOD', 'DAOD055', 'DAOD11']:
            if col_orig in df.columns:
                 dqf_col_orig = f"{col_orig}_DQF"
                 if dqf_col_orig in df.columns: df.loc[df[dqf_col_orig] != 2, col_orig] = np.nan
    elif folder_upper == 'VGT':
        if 'DQF' in df.columns:
            dqf_values = df['DQF'].fillna(0).astype(int).values
            # Bit 7: Space, Bit 2: 전체 품질 (모든 VGT 변수에 적용)
            bit_7 = (dqf_values >> 7) & 1; bit_2 = (dqf_values >> 2) & 1
            invalid_mask = (bit_7 != 0) | (bit_2 != 0)
            # VGT 데이터 변수: NDVI, EVI, FVC (DQF 제외)
            vgt_cols = [c for c in df.columns if c in ['NDVI', 'EVI', 'FVC']]
            df.loc[invalid_mask, vgt_cols] = np.nan
            # 개별 비트 플래그 처리
            if 'NDVI' in df.columns: bit_3 = (dqf_values >> 3) & 1; df.loc[bit_3 != 0, 'NDVI'] = np.nan
            if 'EVI' in df.columns: bit_4 = (dqf_values >> 4) & 1; df.loc[bit_4 != 0, 'EVI'] = np.nan
            if 'FVC' in df.columns: bit_5 = (dqf_values >> 5) & 1; df.loc[bit_5 != 0, 'FVC'] = np.nan
    return df
# --------------------------------------------


# --- 2. GK2A_PAST: Helper 함수들 (process_direct.py에서 복사) ---
PROCESS_FOLDERS = [
    'AII', 'APPS', 'CI', 'CLA', 'CLD', 'CTPS', 'DCOEW', 'FOG', 'LST',
    'LWRAD', 'NCOT', 'QPN', 'RR', 'SAL', 'SWRAD', 'TPW', 'TQPROF', 'VGT'
]
NC_FILE_CACHE = {}
MATCHED_GEO = None 

def scan_nc_files_and_get_kst_dates(sample_dir):
    logger.info(f"[1/6] 📂 NC 파일 스캔 및 KST 날짜 범위 설정: {sample_dir}")
    kst_date_file_map = defaultdict(lambda: defaultdict(list))
    all_utc_dts = set()
    search_path = os.path.join(sample_dir, "**", "*.nc")
    all_nc_files = glob.glob(search_path, recursive=True)
    if not all_nc_files:
        logger.error(f"❌ '{sample_dir}'에서 .nc 파일을 찾을 수 없습니다.")
        return None, None
    pbar = ProgressBar(total=len(all_nc_files), desc="[NC 스캔]")
    files_processed = 0
    skipped_folders = set()
    for filepath in all_nc_files:
        filename = os.path.basename(filepath)
        if filename.endswith('.nc'):
            parts = filename.replace('.nc', '').split('_')
            if len(parts) >= 6:
                folder_name = parts[3].upper()
                utc_timestamp_str = parts[-1]
                if len(utc_timestamp_str) == 12 and utc_timestamp_str.isdigit():
                    if folder_name not in PROCESS_FOLDERS:
                        if folder_name not in skipped_folders: skipped_folders.add(folder_name)
                        pbar.update(1); continue
                    try:
                        utc_dt = datetime.strptime(utc_timestamp_str, "%Y%m%d%H%M")
                        # --- 날짜 범위 체크 ---
                        if not (DATA_START_DATE <= utc_dt <= DATA_END_DATE):
                            pbar.update(1); continue 
                        # ---------------------
                        all_utc_dts.add(utc_dt)
                        kst_dt = utc_dt + timedelta(hours=9)
                        kst_date_str = kst_dt.strftime("%Y-%m-%d")
                        kst_date_file_map[kst_date_str][folder_name].append(filepath)
                        files_processed += 1
                    except ValueError: pass
        pbar.update(1)
    pbar.close()
    
    logger.info(f"  → DEBUG: Total .nc files found by glob: {len(all_nc_files)}")
    logger.info(f"  → DEBUG: Valid .nc files processed: {files_processed}")
    if skipped_folders: logger.info(f"  → DEBUG: Skipped folders: {skipped_folders}")
    
    if not kst_date_file_map:
        logger.error("❌ 처리할 유효한 .nc 파일을 찾지 못했습니다. (파일명 형식 확인 필요)")
        return None, None
    logger.info(f"  → 스캔 완료: {len(kst_date_file_map)}개 KST 날짜 발견.")
    return kst_date_file_map, (min(all_utc_dts), max(all_utc_dts)) if all_utc_dts else (None, None)

def load_matched_geo(matched_geo_path):
    """[수정] 1번 패키지(process 1)의 결과물인 matched_geoid.csv를 *읽어옴*"""
    global MATCHED_GEO
    if MATCHED_GEO is not None: return MATCHED_GEO
    
    logger.info(f"[2/6] 🌍 좌표 매칭 테이블 로드: {matched_geo_path}")
    try:
        matched_geo = pd.read_csv(matched_geo_path)
        if 'geo_id' not in matched_geo.columns:
            matched_geo.rename(columns={'geoId': 'geo_id'}, inplace=True)
        if 'index' not in matched_geo.columns:
             logger.error(f"❌ '{matched_geo_path}' 파일에 'index' 컬럼이 없습니다.")
             return None
        
        MATCHED_GEO = matched_geo[['geo_id', 'geo_lon', 'geo_lat', 'index']] # [수정] matched_distance 제외
        logger.info(f"  → 매칭 테이블 로드 완료 ({len(MATCHED_GEO)}개 지점)")
        return MATCHED_GEO
    except FileNotFoundError:
        logger.error(f"❌ 매칭 테이블 파일을 찾을 수 없습니다: {matched_geo_path}")
        logger.error("  → '프로세스 1' (create_coordinate_mapping.py)을 먼저 실행해야 합니다.")
        return None
    except KeyError as e:
        logger.error(f"❌ 매칭 테이블에 필수 컬럼이 없습니다: {e}")
        return None

def process_nc_file(filepath, folder_name, indices):
    """[그대로] NC 파일 1개를 읽고 모든 처리(Fix)를 수행하여 DataFrame 반환"""
    global NC_FILE_CACHE
    if filepath in NC_FILE_CACHE: return NC_FILE_CACHE[filepath]
    try:
        if folder_name == 'TQPROF':
            ds_full = xr.open_dataset(filepath)
            tqprof_vars = ['Q_profile', 'T_profile', 'quality_flag1', 'quality_flag2']
            available_vars = [v for v in tqprof_vars if v in ds_full.variables]
            if not available_vars: ds_full.close(); return None
            ds = ds_full[available_vars]
            if 'dim_z' in ds.dims: ds = ds.isel(dim_z=51)
            else: logger.warning(f"    - ⚠️ {os.path.basename(filepath)}: dim_z 없음")
            ds_full.close()
        else:
            ds = xr.open_dataset(filepath)
        exclude_vars = ['lat', 'lon', 'latitude', 'longitude', 'time', 'Time', 'x', 'y',
                        'dim_x', 'dim_y', 'dim_z', 'gk2a_imager_projection',
                        'pressure_levels', 'quality_flag3']
        data_vars = [v for v in ds.data_vars if v not in exclude_vars]
        if not data_vars: ds.close(); return None
        df = pd.DataFrame({'index': indices})
        for var in data_vars:
            try:
                var_data = ds[var].values.flatten()
                if len(var_data) > max(indices): df[var] = var_data[indices]
                else: df[var] = np.nan
            except Exception as e_var:
                 logger.warning(f"    - ⚠️ 변수 '{var}' 추출 오류 ({os.path.basename(filepath)}): {e_var}")
                 df[var] = np.nan
        ds.close()

        # [이상치 처리] (원본 이름 기준)
        df = clean_outliers_past(df, folder_name) # [수정] 함수 이름 명시

        # [변수명 매핑] (process_direct.py와 동일)
        if MAPPING_ENABLED:
            df = apply_variable_mapping(df, folder_name) # 'variable_mapping.py' import 필요

        # [KST 변환] (process_direct.py와 동일)
        try:
            timestamp_str = os.path.basename(filepath).split('_')[-1].replace('.nc', '')
            utc_dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
            kst_dt = utc_dt + timedelta(hours=9)
            df['dateTime'] = kst_dt
        except: df['dateTime'] = pd.NaT
        NC_FILE_CACHE[filepath] = df
        return df
    except Exception as e:
        logger.error(f"    - ❌ NC 파일 처리 오류 ({os.path.basename(filepath)}): {e}")
        NC_FILE_CACHE[filepath] = None
        return None

def merge_daily_data(kst_date, file_map_for_date, matched_geo):
    """[그대로] 메모리 최적화 병합 (v3)"""
    logger.info(f"--- {kst_date} 처리 시작 ---")
    base_df_geo_map = matched_geo[['geo_id', 'index']].set_index('index')
    indices = matched_geo['index'].values
    files_to_process = []
    for folder_name, paths in file_map_for_date.items():
        for path in paths: files_to_process.append((path, folder_name))
    if not files_to_process:
        logger.warning("  - 처리할 NC 파일 없음")
        return None
    pbar = ProgressBar(total=len(files_to_process), desc=f"    [NC 처리: {kst_date}]")
    temp_processed_data = defaultdict(list)
    for filepath, folder_name in files_to_process:
        df_processed_with_index = process_nc_file(filepath, folder_name, indices)
        if df_processed_with_index is not None and not df_processed_with_index.empty and 'index' in df_processed_with_index.columns and 'dateTime' in df_processed_with_index.columns:
            df_processed = pd.merge(df_processed_with_index, base_df_geo_map, left_on='index', right_index=True, how='left')
            if 'index' in df_processed.columns: df_processed.drop(columns=['index'], inplace=True)
            df_processed.dropna(subset=['dateTime'], inplace=True)
            if not df_processed.empty:
                temp_processed_data[folder_name].append(df_processed)
        pbar.update(1)
    pbar.close()
    del df_processed_with_index, df_processed
    gc.collect()
    if not temp_processed_data:
        logger.warning("  - 병합할 처리된 변수 데이터 없음")
        return None
    logger.info(f"    → 🛠️ 변수별 데이터 정리 중...")
    processed_data_merged = {}
    for folder_name, df_list in temp_processed_data.items():
        if df_list:
            df_concat = pd.concat(df_list, ignore_index=True)
            df_concat.set_index(['geo_id', 'dateTime'], inplace=True)
            df_concat = df_concat[~df_concat.index.duplicated(keep='first')]
            processed_data_merged[folder_name] = df_concat
    del temp_processed_data
    gc.collect()
    all_var_dfs = []
    for folder_name, df_var in processed_data_merged.items():
         cols_to_keep = [c for c in df_var.columns if c not in ['geo_id', 'dateTime']]
         if cols_to_keep: all_var_dfs.append(df_var[cols_to_keep])
    logger.info(f"    → 🔗 변수 병합 중... (총 {len(all_var_dfs)}개 변수)")
    combined_vars_df = None
    if len(all_var_dfs) == 0:
         logger.warning("  - 병합할 변수 데이터 없음")
         return None
    elif len(all_var_dfs) == 1:
        combined_vars_df = all_var_dfs[0]
    else:
        combined_vars_df = all_var_dfs[0]
        for i in range(1, len(all_var_dfs)):
            combined_vars_df = combined_vars_df.join(all_var_dfs[i], how='outer')
    if combined_vars_df is None or combined_vars_df.empty:
        logger.warning("  - 최종 병합된 변수 데이터 없음")
        return None
    combined_vars_df.reset_index(inplace=True)
    del all_var_dfs, processed_data_merged
    gc.collect()
    geo_coords = matched_geo[['geo_id', 'geo_lon', 'geo_lat']].drop_duplicates(subset='geo_id')
    final_merged = pd.merge(combined_vars_df, geo_coords, on='geo_id', how='left')
    logger.info(f"    → ✓ 병합 완료 (총 {len(final_merged)}행)")
    final_merged.dropna(subset=['dateTime'], inplace=True)
    return final_merged

def apply_special_rules_and_format(df):
    """[그대로] (작업 7, 8) SAL/VGT 보간 및 최종 컬럼 정렬"""
    existing_fill_cols = [col for col in DAILY_FILL_COLUMNS if col in df.columns]
    if existing_fill_cols:
        logger.info("    → 🔄 SAL/VGT 00시 값 채우기 적용 중...")
        df.sort_values(by=['geo_id', 'dateTime'], inplace=True)
        df[existing_fill_cols] = df.groupby('geo_id')[existing_fill_cols].transform('ffill')
        df[existing_fill_cols] = df.groupby('geo_id')[existing_fill_cols].transform('bfill')
    logger.info("    → 🏛️ 최종 포맷(88개 컬럼)으로 정렬 중...")
    if 'geo_id' in df.columns:
        df.rename(columns={'geo_id': 'geoId'}, inplace=True)
    df_final = pd.DataFrame(columns=FINAL_COLUMNS)
    for col in FINAL_COLUMNS:
        if col in df.columns: df_final[col] = df[col]
        else: df_final[col] = pd.NA
    return df_final

def save_hourly_files_direct(df, hourly_output_dir):
    """[그대로] (작업 9) 최종 DF를 KST 날짜/시간 기준으로 시간별 파일 저장"""
    logger.info(f"    → 💾 시간별 파일 저장 중 (geoId별 마지막 값 유지)...")
    saved_count = 0
    if not pd.api.types.is_datetime64_any_dtype(df['dateTime']):
         df['dateTime'] = pd.to_datetime(df['dateTime'])
    df['kst_date_str'] = df['dateTime'].dt.strftime('%Y-%m-%d')
    groups = df.groupby(['kst_date_str', df['dateTime'].dt.hour])
    pbar = ProgressBar(total=len(groups), desc="    [파일 저장]")
    for (date_str, hour), daily_group in groups:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            year, month, day = dt.year, dt.month, dt.day
        except ValueError:
            logger.warning(f"    - ❌ 날짜 형식 오류: {date_str}. 저장할 수 없습니다.")
            pbar.update(1); continue
        hourly_df_with_duplicates = daily_group.copy()
        if hourly_df_with_duplicates.empty: pbar.update(1); continue
        hourly_df_with_duplicates.sort_values(by='dateTime', ascending=True, inplace=True)
        hourly_df = hourly_df_with_duplicates.groupby('geoId').last().reset_index()
        if hourly_df.empty: pbar.update(1); continue
        hour_str = f"{hour:02d}"
        output_path = os.path.join(hourly_output_dir, "KOMPSAT_LE2", str(year), str(month), str(day))
        os.makedirs(output_path, exist_ok=True)
        filename = f"KOMPSAT_LE2_{date_str}_{hour_str}.csv"
        filepath = os.path.join(output_path, filename)
        if 'geoId' in hourly_df.columns:
            hourly_df['geoId'] = hourly_df['geoId'].astype('Int64')
        if 'dateTime' in hourly_df.columns and pd.api.types.is_datetime64_any_dtype(hourly_df['dateTime']):
             hourly_df['dateTime'] = hourly_df['dateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        hourly_df.drop(columns=['kst_date_str'], inplace=True, errors='ignore')
        cols_ordered = [col for col in FINAL_COLUMNS if col in hourly_df.columns]
        hourly_df = hourly_df[cols_ordered]
        try:
            hourly_df.to_csv(filepath, index=False)
            saved_count += 1
        except Exception as e:
            logger.error(f"    - ❌ 파일 저장 오류 ({filename}): {e}")
        pbar.update(1)
    pbar.close()
    logger.info(f"    → ✓ {saved_count}개의 시간별 파일 저장 완료.")
    return saved_count

def check_if_day_already_processed(kst_date, hourly_output_dir):
    """[그대로] --resume 기능"""
    try:
        dt = datetime.strptime(kst_date, "%Y-%m-%d")
        year, month, day = dt.year, dt.month, dt.day
    except ValueError: return False
    output_path = os.path.join(hourly_output_dir, "KOMPSAT_LE2", str(year), str(month), str(day))
    if os.path.exists(output_path):
        return True
    return False
# ----------------------------------------------------

# ==============================================================================
# [모드 2] API (.csv) 처리 로직
# ==============================================================================

def check_missing(df, out):
    save_dir = Path('../../Data/1.outliers_rm/missing')
    save_dir.mkdir(parents=True, exist_ok=True)

    overall_path = save_dir / f'{out}_overall.csv'
    hourly_path = save_dir / f'{out}_hourly.csv'

    exclude_cols = ['dateTime', 'hour', 'geoId', 'dateTime', 'year', 'month', 'day', 'st_lon', 'st_lat', 'geo_lon', 'geo_lat']
    target_cols = [c for c in df.columns if c not in exclude_cols]

    # 1) 전체 누적 결측
    current_total_rows = len(df)
    current_missing = df[target_cols].isna().sum()

    overall_current = pd.DataFrame({
        'column': current_missing.index,
        'missing_count': current_missing.values
    })
    overall_current['total_rows'] = current_total_rows

    if overall_path.exists():
        overall_prev = pd.read_csv(overall_path)

        overall_merged = pd.merge(
            overall_prev[['column', 'missing_count', 'total_rows']],
            overall_current,
            on='column',
            how='outer',
            suffixes=('_prev', '_curr')
        ).fillna(0)

        overall_merged['missing_count'] = (
            overall_merged['missing_count_prev'] + overall_merged['missing_count_curr']
        )
        overall_merged['total_rows'] = (
            overall_merged['total_rows_prev'] + overall_merged['total_rows_curr']
        )

        overall_result = overall_merged[['column', 'missing_count', 'total_rows']]
    else:
        overall_result = overall_current[['column', 'missing_count', 'total_rows']]

    overall_result['missing_rate'] = overall_result['missing_count'] / overall_result['total_rows']
    overall_result['missing_percent'] = overall_result['missing_rate'] * 100
    overall_result.to_csv(overall_path, index=False)

    # 2) 시간대별 누적 결측
    # 각 hour별 행 수
    hourly_total = df.groupby('hour').size().rename('total_rows').reset_index()

    # 각 hour별 각 column의 결측 수
    hourly_missing = (
        df.groupby('hour')[target_cols]
          .apply(lambda x: x.isna().sum())
          .reset_index()
    )

    # wide -> long
    hourly_missing_long = hourly_missing.melt(
        id_vars='hour',
        var_name='column',
        value_name='missing_count'
    )

    # hour별 total_rows 붙이기
    hourly_current = pd.merge(hourly_missing_long, hourly_total, on='hour', how='left')

    if hourly_path.exists():
        hourly_prev = pd.read_csv(hourly_path)

        hourly_merged = pd.merge(
            hourly_prev[['hour', 'column', 'missing_count', 'total_rows']],
            hourly_current,
            on=['hour', 'column'],
            how='outer',
            suffixes=('_prev', '_curr')
        ).fillna(0)

        hourly_merged['missing_count'] = (
            hourly_merged['missing_count_prev'] + hourly_merged['missing_count_curr']
        )
        hourly_merged['total_rows'] = (
            hourly_merged['total_rows_prev'] + hourly_merged['total_rows_curr']
        )

        hourly_result = hourly_merged[['hour', 'column', 'missing_count', 'total_rows']]
    else:
        hourly_result = hourly_current[['hour', 'column', 'missing_count', 'total_rows']]

    hourly_result['missing_rate'] = hourly_result['missing_count'] / hourly_result['total_rows']
    hourly_result['missing_percent'] = hourly_result['missing_rate'] * 100
    hourly_result = hourly_result.sort_values(['column', 'hour']).reset_index(drop=True)

    hourly_result.to_csv(hourly_path, index=False)

def outliers_quantile(df, data_cols):
    for col in data_cols:
        lower = df[col].quantile(0.001)
        upper = df[col].quantile(0.999)
        mask = (df[col] < lower) | (df[col] > upper)
        df.loc[mask, col] = np.nan
    return df

def clean_gk2a_outliers_api(df):
    """[API 데이터용] (API 컬럼명 기준)"""
    logger.info("  → GK2A API 이상치 처리 중 (API 컬럼명 기준)...")
    invalid_values = [-99, 999, -999, 65535]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data_cols = [c for c in numeric_cols if '_dqf' not in c and '_flag' not in c and c not in ['geoId', 'geo_lon', 'geo_lat', 'st_lon','st_lat', 'dateTime']]

    for col in data_cols:
        if col in df.columns: df[col] = df[col].replace(invalid_values, np.nan)

    check_missing(df, 'missing_gk2a_raw')
    df = outliers_quantile(df, data_cols)
    check_missing(df, 'missing_gk2a_outliers_rm')
            
    logger.info("  → GK2A API 이상치 처리 완료.")
    return df

def clean_airkorea_outliers_api(df):
    logger.info("  → AIRKOREA 이상치 처리 중...")
    
    # (10/29 톡 + 노션 2.2 로직)
    # 1. 0 이하의 값을 NA로 대체
    value_cols = ['so2Value', 'coValue', 'o3Value', 'no2Value', 'pm10Value', 'pm25Value']
    for col in value_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] <= 0, col] = np.nan
            
    # 2. Flag 값 존재 시 NA로 대체
    flag_pairs = {
        'so2Value': 'so2Flag', 'coValue': 'coFlag', 'o3Value': 'o3Flag',
        'no2Value': 'no2Flag', 'pm10Value': 'pm10Flag', 'pm25Value': 'pm25Flag'
    }
    
    for value_col, flag_col in flag_pairs.items():
        if value_col in df.columns and flag_col in df.columns:
            # '0' (int 또는 str)과 NaN(결측)이 아닌 모든 값을 "Bad"로 간주
            # (예: '통신장애', 1, 9 등)
            flag_values = df[flag_col].fillna('0').astype(str)
            bad_flag_mask = (flag_values != '0')
            df.loc[bad_flag_mask, value_col] = np.nan
            
    logger.info("  → AIRKOREA 이상치 처리 완료.")
    return df

def clean_odam_outliers_api(df):
    """
    [API 데이터용] ODAM(기상청 실황) 이상치 처리 (API 컬럼명 기준)
    (노션 2.3 규칙 / 실제 CSV 컬럼(소문자) 기준으로 수정됨)
    """
    logger.info("  → ODAM API 이상치 처리 중 (API 컬럼명 기준)...")
    
    # 1. 처리 대상 컬럼 리스트 정의 (소문자로 수정)
    odam_cols = ['t1h', 'rn1', 'reh', 'vec', 'uuu', 'vvv', 'wsd', 'pty']
    
    # 2. (공통) 모든 대상 컬럼을 숫자형으로 변환 (숫자 아닌 값은 NaN으로)
    for col in odam_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.debug(f"    - ODAM 경고: '{col}' 컬럼이 df에 없습니다.")

    # 3. (공통) 999 / -999 결측값 처리
    invalid_values = [999, -999, -99]
    for col in odam_cols:
         if col in df.columns:
            df[col] = df[col].replace(invalid_values, np.nan)

    check_missing(df, 'missing_odam_raw')

    # 4. 개별 변수 규칙 적용 (컬럼명 모두 소문자로 수정)

    # T1H (기온): -40 ~ 60 C 범위 마스킹
    if 't1h' in df.columns:
        df.loc[(df['t1h'] < -40) | (df['t1h'] > 60), 't1h'] = np.nan
        
    # RN1, REH, VEC (강수, 습도, 풍향): 음수 범위 삭제
    for col in ['rn1', 'reh', 'vec']:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
            
    # UUU, VVV (풍속 U/V): -99/99, -53.03 삭제
    for col in ['uuu', 'vvv']:
        if col in df.columns:
            df[col] = df[col].replace([-99, -53.03], np.nan)

    # WSD (풍속): 0, 99 삭제
    if 'wsd' in df.columns:
        df['wsd'] = df['wsd'].replace([0, 99], np.nan)
        
    # PTY (강수형태): 0 ~ 7 제외 값 삭제
    if 'pty' in df.columns:
        # PTY는 0~7 사이의 정수 코드값으로 가정
        df.loc[(df['pty'] < 0) | (df['pty'] > 7), 'pty'] = np.nan

    df = outliers_quantile(df, odam_cols)

    check_missing(df, 'missing_odam_outliers_rm')

    logger.info("  → ODAM API 이상치 처리 완료.")
    return df

def clean_gems_outliers_api(df):
    """
    [API 데이터용] GEMS 이상치 처리 (API 컬럼명 기준)
    (GEMS DQF 규칙이 미전달된 상태이므로, 플래그 != 0 인 경우를 기본 결측 처리로 가정)
    """
    logger.info("  → GEMS API 이상치 처리 중 (기본 플래그 규칙 적용)...")
    
    # 1. 공통 결측값 처리 (GK2A와 동일하게)
    invalid_values = [99, -99, 999, -999, 65535]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns: df[col] = df[col].replace(invalid_values, np.nan)
        
    # 2. DQF/Flag 기반 결측 처리 (플래그 != 0 이면 데이터 NaN 처리로 가정)
    
    # [NO2] - groundPixelQualityFlags, finalAlgorithmFlags
    # 'finalAlgorithmFlags'가 NO2, AOD 등에 적용되는 주요 플래그로 가정
    no2_cols = ['columnAmountNo2', 'columnAmountNo2Strat', 'columnAmountNo2Trop']
    if 'finalAlgorithmFlags' in df.columns:
        mask = (df['finalAlgorithmFlags'] != 0)
        df.loc[mask, [c for c in no2_cols if c in df.columns]] = np.nan
         
    # [SO2] - algorithmQualityFlag
    so2_cols = ['columnAmountSo2']
    if 'algorithmQualityFlag' in df.columns:
        mask = (df['algorithmQualityFlag'] != 0)
        df.loc[mask, [c for c in so2_cols if c in df.columns]] = np.nan
        
    # [AOD/Aerosol] - finalAlgorithmFlags (NO2와 플래그 공유)
    aero_cols = ['finalAerosolOpticalDepth', 'finalAerosolLayerHeight', 'aerosolOpticalDepth']
    if 'finalAlgorithmFlags' in df.columns:
        mask = (df['finalAlgorithmFlags'] != 0)
        df.loc[mask, [c for c in aero_cols if c in df.columns]] = np.nan
        
    # [O3/Ozone] - tocQualityFlag
    o3_cols = ['totalOzoneColumnUvi', 'columnAmountO3']
    if 'tocQualityFlag' in df.columns:
         mask = (df['tocQualityFlag'] != 0)
         df.loc[mask, [c for c in o3_cols if c in df.columns]] = np.nan
         
    # [Cloud] - groundPixelQualityFlags (Cloud도 픽셀 퀄리티에 영향 받음)
    cloud_cols = ['cloudFraction', 'cloudPressure', 'cloudCentroidPressure', 'effectiveCloudFraction', 'cloudOpticalDepth']
    if 'groundPixelQualityFlags' in df.columns:
        mask = (df['groundPixelQualityFlags'] != 0)
        df.loc[mask, [c for c in cloud_cols if c in df.columns]] = np.nan
    
    # [기타] - dataIndexFlag (전반적인 데이터 인덱스 유효성)
    data_cols = [c for c in df.columns if c not in ['geoId', 'geo_lon', 'geo_lat', 'dateTime', 'st_lon', 'st_lat'] and not c.lower().endswith('flag')]
    if 'dataIndexFlag' in df.columns:
        mask = (df['dataIndexFlag'] != 0)
        df.loc[mask, [c for c in data_cols if c in df.columns]] = np.nan
        
    logger.info("  → GEMS API 이상치 처리 완료.")
    return df

def process_api_data(input_dir, output_dir, data_type, resume, force):
    """
    [API 모드 - v2 수정]
    API CSV *폴더*를 입력받아 이상치 처리 후, 시간별로 분할하여 저장
    """
    logger.info("="*60)
    logger.info(f"[프로세스 2: {data_type}] 시작")
    logger.info(f"  - API 입력 폴더: {input_dir}")
    logger.info(f"  - API 출력 폴더: {output_dir}")
    logger.info("="*60)
    
    # 1. 입력 폴더에서 CSV 파일 스캔
    # (API 데이터가 어떤 형식(일별/월별)으로 저장될지 모르므로, 일단 하위 모든 CSV를 스캔)
    search_path = os.path.join(input_dir, "**", "*.csv")
    all_csv_files = glob.glob(search_path, recursive=True)
    
    if not all_csv_files:
        logger.error(f"❌ '{input_dir}'에서 API CSV 파일을 찾을 수 없습니다.")
        return

    logger.info(f"  → {len(all_csv_files)}개 API CSV 파일 발견. 처리 시작...")
    pbar = ProgressBar(total=len(all_csv_files), desc=f"[API 처리: {data_type}]")
    total_files_saved = 0

    for input_file in all_csv_files:
        try:
            # 2. CSV 로드
            logger.info(f"  → 파일 로드 중: {os.path.basename(input_file)}")
            # low_memory=False 추가하여 DtypeWarning 방지
            df = pd.read_csv(input_file, low_memory=False)

            # [컬럼명 통일] snake_case/camelCase 혼용 대응 (GK2A, ODAM 등)
            rename_map = {}
            if 'date_time' in df.columns: rename_map['date_time'] = 'dateTime'
            if 'dataTime' in df.columns: rename_map['dataTime'] = 'dateTime'
            if 'geo_id' in df.columns: rename_map['geo_id'] = 'geoId'
            if 'geoLon' in df.columns: rename_map['geoLon'] = 'geo_lon'
            if 'geoLat' in df.columns: rename_map['geoLat'] = 'geo_lat'
            if 'stLon' in df.columns: rename_map['stLon'] = 'st_lon'
            if 'stLat' in df.columns: rename_map['stLat'] = 'st_lat'
            
            if rename_map:
                df.rename(columns=rename_map, inplace=True)
            
            if 'dateTime' not in df.columns or 'geoId' not in df.columns:
                logger.warning(f"    - ⚠️ '{os.path.basename(input_file)}'에 'dateTime' 또는 'geoId' 컬럼이 없어 건너뜁니다.")
                pbar.update(1); continue
            
            df['dateTime'] = pd.to_datetime(df['dateTime'])
            df['year'] = df['dateTime'].dt.year
            df['month'] = df['dateTime'].dt.month
            df['day'] = df['dateTime'].dt.day
            df['hour'] = df['dateTime'].dt.hour

            # 3. 데이터 타입에 따라 분기
            if data_type == 'GK2A_API':
                cleaned_df = clean_gk2a_outliers_api(df)
            elif data_type == 'AIRKOREA_API':
                cleaned_df = clean_airkorea_outliers_api(df)
            elif data_type == 'ODAM_API':
                cleaned_df = clean_odam_outliers_api(df)
            elif data_type == 'GEMS_API': # <--- 이 부분 추가
                cleaned_df = clean_gems_outliers_api(df)
            else:
                logger.error(f"❌ 지원하지 않는 API 데이터 타입입니다: {data_type}")
                pbar.update(1); continue
                
            # 4. [수정] KST 시간별로 분할하여 저장 (GK2A_PAST 모드의 save_hourly... 함수와 유사)
            logger.info(f"  → 시간별 파일 저장 중...")
            
            if not pd.api.types.is_datetime64_any_dtype(cleaned_df['dateTime']):
                 cleaned_df['dateTime'] = pd.to_datetime(cleaned_df['dateTime'])
            
            cleaned_df['kst_date_str'] = cleaned_df['dateTime'].dt.strftime('%Y-%m-%d')
            cleaned_df['kst_hour'] = cleaned_df['dateTime'].dt.hour

            saved_count_for_file = 0
            
            # API 데이터 소스 이름 (출력 폴더명용)
            # (예: GK2A_API -> KOMPSAT_LE2, AIRKOREA_API -> AIRKOREA)
            if data_type == 'GK2A_API':
                output_source_name = "KOMPSAT_LE2"
            elif data_type == 'AIRKOREA_API':
                output_source_name = "AIRKOREA"
            elif data_type == 'ODAM_API':
                output_source_name = "WEATHER_ODAM"
            elif data_type == 'GEMS_API': # <--- 이 부분 추가
                output_source_name = "GEMS"
            else:
                output_source_name = data_type # GEMS 등

            for (date_str, hour), hourly_df_group in cleaned_df.groupby(['kst_date_str', 'kst_hour']):
                if hourly_df_group.empty: continue
                hourly_df = hourly_df_group.copy()
                
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    year, month, day = str(dt.year), str(dt.month), str(dt.day)
                except ValueError: continue

                hour_str = f"{hour:02d}"
                # [수정] GK2A_PAST와 동일한 HOURLY_DATA 구조로 저장
                output_path = os.path.join(output_dir, output_source_name, year, str(int(month)), str(day))
                os.makedirs(output_path, exist_ok=True)
                
                # 파일명 형식 (예: AIRKOREA_2025-03-15_01.csv)
                filename = f"{output_source_name}_{date_str}_{hour_str}.csv"
                filepath = os.path.join(output_path, filename)
                
                # --resume 처리
                if resume and not force and os.path.exists(filepath):
                    continue # 이미 있으면 건너뛰기

                if 'geoId' in hourly_df.columns:
                    hourly_df['geoId'] = hourly_df['geoId'].astype('Int64')
                hourly_df['dateTime'] = hourly_df['dateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                hourly_df.drop(columns=['kst_date_str', 'kst_hour'], inplace=True, errors='ignore')

                hourly_df.to_csv(filepath, index=False)
                saved_count_for_file += 1
            
            logger.info(f"  → ✓ {os.path.basename(input_file)} 처리 완료 ({saved_count_for_file}개 시간별 파일 저장)")
            total_files_saved += saved_count_for_file

        except FileNotFoundError:
            logger.error(f"❌ 입력 파일 없음: {input_file}")
        except Exception as e:
            logger.exception(f"❌ {os.path.basename(input_file)} 처리 중 치명적인 오류 발생")
        
        pbar.update(1)

    pbar.close()
    logger.info("="*60)
    logger.info(f"[프로세스 2: {data_type}] 작업 완료")
    logger.info(f"  - 총 {total_files_saved}개의 시간별 파일이 생성/업데이트되었습니다.")
    logger.info("="*60)

# -----------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[프로세스 2] 과거 데이터(.nc) 또는 API 데이터(.csv)의 이상치를 처리합니다.",
        epilog="""
사용 예시:

  # 1. [과거] GK2A 과거 데이터(.nc) 처리 (1&2 통합) -> HOURLY_DATA 생성
  python run_process_2_outliers.py \\
    --data-type GK2A_PAST \\
    --input-sample /path/to/GK2A_KMAHDD \\
    --input-match-table /path/to/matched_geoid.csv \\
    --output /path/to/HOURLY_DATA \\
    --resume

  # 2. [API] AIRKOREA API 데이터(.csv) 처리 (2단계) -> HOURLY_DATA에 추가
  python run_process_2_outliers.py \\
    --data-type AIRKOREA_API \\
    --input-dir /path/to/API_AIRKOREA_raw_csvs \\
    --output /path/to/HOURLY_DATA
"""
    )
    
    # --- [수정] 인자 그룹화 및 변경 ---
    parser.add_argument( '--data-type', '-t', type=str, required=True, 
                         choices=['GK2A_PAST', 'GK2A_API', 'AIRKOREA_API', 'ODAM_API', 'GEMS_API'],
                         help='처리할 데이터 소스 타입')
    parser.add_argument( '--output', '-o', type=str, required=True, help='[공통] 처리된 시간별 CSV를 저장할 *최상위* 폴더 (예: HOURLY_DATA)')
    parser.add_argument( '--resume', action='store_true', help='[공통] 중단된 작업 재개 (이미 처리된 시간별 파일 건너뛰기)')
    parser.add_argument( '--force', action='store_true', help='[공통] 기존 파일 덮어쓰기 (--resume 옵션 무시)')

    # 과거 데이터 (.nc) 처리용 인자
    past_group = parser.add_argument_group('Past Data (.nc) Arguments (GK2A_PAST)')
    past_group.add_argument( '--input-sample', type=str, help='[GK2A_PAST] 원본 .nc 파일이 있는 sample 디렉토리')
    past_group.add_argument( '--input-match-table', type=str, help="[GK2A_PAST] '프로세스 1'에서 생성된 matched_geoid.csv 파일 경로")
    
    # API 데이터 (.csv) 처리용 인자
    api_group = parser.add_argument_group('API Data (.csv) Arguments (GK2A_API, AIRKOREA_API, ...)')
    api_group.add_argument( '--input-dir', type=str, help='[API] 이상치 처리가 필요한 원본 API CSV가 있는 *폴더* 경로')
    
    args = parser.parse_args()
    
    # --- 로거 설정 (파일/콘솔 분리) ---
    log_dir = os.path.join(args.output, "logs") # [수정] 공통 output 폴더에 로그 저장
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_process_2_{args.data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger(log_file)
    # -----------------------------------

    
    # --- 데이터 타입에 따라 실행 ---
    if args.data_type == 'GK2A_PAST':
        # [모드 1] (process_direct_hourly.py와 동일한 로직 실행)
        if not args.input_sample or not args.input_match_table:
            logger.error("❌ 'GK2A_PAST' 타입은 --input-sample, --input-match-table 인자가 반드시 필요합니다.")
            sys.exit(1)
        if not MAPPING_ENABLED:
            logger.error("❌ 'GK2A_PAST' 타입은 'variable_mapping.py' 파일이 반드시 필요합니다.")
            sys.exit(1)
            
        try:
            # (process_direct_hourly.py의 main 함수 로직이 여기로...)
            logger.info("="*70)
            logger.info("🛰️  [프로세스 1&2 통합: GK2A_PAST] 시작")
            logger.info(f"  - NC 입력: {args.input_sample}")
            logger.info(f"  - 매칭 테이블: {args.input_match_table}")
            logger.info(f"  - 최종 출력: {args.output}")
            if args.resume: logger.info("  - 모드: 🔄 Resume")
            logger.info("="*70)
            
            kst_date_file_map, _ = scan_nc_files_and_get_kst_dates(args.input_sample)
            if kst_date_file_map is None: sys.exit(1)
            matched_geo = load_matched_geo(args.input_match_table)
            if matched_geo is None: sys.exit(1)

            logger.info(f"[3/6] 🔄 {len(kst_date_file_map)}개 KST 날짜 처리를 시작합니다...")
            main_pbar = ProgressBar(total=len(kst_date_file_map), desc="[전체 진행 (KST 날짜)]")
            total_files_saved = 0
            skipped_dates = 0

            for kst_date in sorted(kst_date_file_map.keys()):
                if args.resume and not args.force:
                    if check_if_day_already_processed(kst_date, args.output):
                        logger.debug(f"{kst_date}: 건너뛰기 (폴더가 이미 존재함)")
                        main_pbar.update(1); skipped_dates += 1; continue

                file_map_for_date = kst_date_file_map[kst_date]
                daily_merged_df = merge_daily_data(kst_date, file_map_for_date, matched_geo)
                if daily_merged_df is None or daily_merged_df.empty:
                    logger.warning(f"    - ❌ {kst_date}: 병합된 데이터 없음. 건너뜁니다.")
                    main_pbar.update(1); NC_FILE_CACHE.clear(); gc.collect(); continue
                final_daily_df = apply_special_rules_and_format(daily_merged_df)
                saved_count = save_hourly_files_direct(final_daily_df, args.output)
                total_files_saved += saved_count
                del daily_merged_df, final_daily_df
                NC_FILE_CACHE.clear(); gc.collect()
                main_pbar.update(1)

            main_pbar.close()
            logger.info("="*70)
            logger.info("🎉 [GK2A_PAST] 작업이 완료되었습니다.")
            logger.info(f"  - 총 {total_files_saved}개의 시간별 파일이 생성/업데이트되었습니다.")
            if args.resume: logger.info(f"  - 총 {skipped_dates}개 날짜를 건너뛰었습니다.")
            logger.info(f"최종 산출물은 {args.output} 에서 확인하세요.")
            logger.info("="*70)

        except Exception as e:
             logger.exception("💥 [GK2A_PAST] 스크립트 실행 중 치명적인 오류 발생")
             print(f"\n❌ 심각한 오류 발생: {e}")
             if log_file: print(f"상세 내용은 로그 파일을 확인하세요: {log_file}")
    
    elif args.data_type in ['GK2A_API', 'AIRKOREA_API', 'ODAM_API', 'GEMS_API']: # <--- 'GEMS_API' 추가
        # [모드 2] (API 데이터 처리)
        if not args.input_dir:
             logger.error(f"❌ '{args.data_type}' 타입은 --input-dir 인자가 반드시 필요합니다.")
             sys.exit(1)
        # [수정] process_api_data 함수 호출
        process_api_data(args.input_dir, args.output, args.data_type, args.resume, args.force)
        
    else:
        logger.error(f"❌ 알 수 없는 데이터 타입입니다: {args.data_type}")