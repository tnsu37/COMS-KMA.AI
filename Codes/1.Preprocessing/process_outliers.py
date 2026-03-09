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


def clean_gk2a_outliers_api(df):
    """[API 데이터용] (API 컬럼명 기준)"""
    logger.info("  → GK2A API 이상치 처리 중 (API 컬럼명 기준)...")
    invalid_values = [99, -99, 999, -999, 65535]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data_cols = [c for c in numeric_cols if '_dqf' not in c and '_flag' not in c and c not in ['geoId', 'geo_lon', 'geo_lat']]
    for col in data_cols:
        if col in df.columns: df[col] = df[col].replace(invalid_values, np.nan)
    
    # (AII, TQPROF, TPW, LST, FOG, DCOEW, CTPS, CLA, NCOT, SWRAD, SAL, LWRAD, APPS... 로직 100% 동일)
    if 'aii_dqf1' in df.columns and 'aii_dqf2' in df.columns:
        aii_cols = ['aii_cape', 'aii_ki', 'aii_li', 'aii_si', 'aii_tti']
        mask = (df['aii_dqf1'] != 0) | (df['aii_dqf2'] != 1)
        existing_aii_cols = [c for c in aii_cols if c in df.columns]
        df.loc[mask, existing_aii_cols] = np.nan
    if 'tqprof_dqf1' in df.columns and 'tqprof_dqf2' in df.columns:
        tqprof_cols = ['tqprof_t', 'tqprof_q']
        mask = (df['tqprof_dqf1'] != 0) | (df['tqprof_dqf2'] != 1)
        existing_tqprof_cols = [c for c in tqprof_cols if c in df.columns]
        df.loc[mask, existing_tqprof_cols] = np.nan
    if 'tpw_dqf1' in df.columns and 'tpw_dqf2' in df.columns:
        tpw_cols = ['tpw', 'tpw_low', 'tpw_mid', 'tpw_high']
        mask = (df['tpw_dqf1'] != 0) | (df['tpw_dqf2'] != 1)
        existing_tpw_cols = [c for c in tpw_cols if c in df.columns]
        df.loc[mask, existing_tpw_cols] = np.nan
    if 'lst_dqf' in df.columns and 'lst' in df.columns:
        df.loc[df['lst_dqf'] != 0, 'lst'] = np.nan
    if 'fog_dqf' in df.columns and 'fog' in df.columns:
        df.loc[df['fog_dqf'] != 0, 'fog'] = np.nan
    if 'dcoew_dqf1' in df.columns:
        dcoew_cols = ['dcoew_thickness', 'dcoew_radius', 'dcoew_liquid_path']
        existing_dcoew_cols = [c for c in dcoew_cols if c in df.columns]
        df.loc[df['dcoew_dqf1'] != 0, existing_dcoew_cols] = np.nan
    if 'ctps_dqf1' in df.columns:
        ctps_cols = ['ctps_cp', 'ctps_cth', 'ctps_ctp', 'ctps_ctt']
        existing_ctps_cols = [c for c in ctps_cols if c in df.columns]
        df.loc[df['ctps_dqf1'] != 0, existing_ctps_cols] = np.nan
    if 'cla_cloud_fraction_dqf' in df.columns and 'cla_cloud_fraction' in df.columns:
        df.loc[df['cla_cloud_fraction_dqf'] != 0, 'cla_cloud_fraction'] = np.nan
    if 'cla_type_dqf' in df.columns and 'cla_type' in df.columns:
        df.loc[df['cla_type_dqf'] != 0, 'cla_type'] = np.nan
    if 'ncot_dqf' in df.columns and 'ncot' in df.columns:
        df.loc[df['ncot_dqf'] != 0, 'ncot'] = np.nan
    if 'swrad_dqf1' in df.columns: 
        swrad_cols = ['swrad_absorbed', 'swrad_downward']
        existing_swrad_cols = [c for c in swrad_cols if c in df.columns]
        df.loc[df['swrad_dqf1'] != 1, existing_swrad_cols] = np.nan
    if 'sal_dqf1' in df.columns:
        sal_bsa_cols = [c for c in df.columns if c.startswith('sal_bsa')]
        df.loc[df['sal_dqf1'] != 1, sal_bsa_cols] = np.nan
    if 'sal_dqf2' in df.columns:
        sal_wsa_cols = [c for c in df.columns if c.startswith('sal_wsa')]
        df.loc[df['sal_dqf2'] != 1, sal_wsa_cols] = np.nan
    if 'lwrad_dqf1' in df.columns: 
        lwrad_cols = ['lwrad_downward', 'lwrad_upward']
        existing_lwrad_cols = [c for c in lwrad_cols if c in df.columns]
        df.loc[df['lwrad_dqf1'] != 1, existing_lwrad_cols] = np.nan
    if 'apps_aep_dqf' in df.columns and 'apps_aep' in df.columns:
        df.loc[df['apps_aep_dqf'] != 2, 'apps_aep'] = np.nan
    if 'apps_aod_dqf' in df.columns and 'apps_aod' in df.columns:
        df.loc[df['apps_aod_dqf'] != 2, 'apps_aod'] = np.nan
    if 'apps_daod055_dqf' in df.columns and 'apps_daod055' in df.columns:
        df.loc[df['apps_daod055_dqf'] != 2, 'apps_daod055'] = np.nan
    if 'apps_daod11_dqf' in df.columns and 'apps_daod11' in df.columns:
        df.loc[df['apps_daod11_dqf'] != 2, 'apps_daod11'] = np.nan
    if 'vgt_dqf1' in df.columns:
        dqf_values = df['vgt_dqf1'].fillna(0).astype(int).values
        bit_7 = (dqf_values >> 7) & 1; bit_2 = (dqf_values >> 2) & 1
        bit_3 = (dqf_values >> 3) & 1; bit_4 = (dqf_values >> 4) & 1
        invalid_mask = (bit_7 != 0) | (bit_2 != 0)
        vgt_cols = ['vgt_ndvi', 'vgt_evi']
        existing_vgt_cols = [c for c in vgt_cols if c in df.columns]
        df.loc[invalid_mask, existing_vgt_cols] = np.nan
        if 'vgt_ndvi' in df.columns: df.loc[bit_3 != 0, 'vgt_ndvi'] = np.nan
        if 'vgt_evi' in df.columns: df.loc[bit_4 != 0, 'vgt_evi'] = np.nan
            
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
    invalid_values = [999, -999]
    for col in odam_cols:
         if col in df.columns:
            df[col] = df[col].replace(invalid_values, np.nan)

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
            df[col] = df[col].replace([-99, 99, -53.03], np.nan)

    # WSD (풍속): 0, 99 삭제
    if 'wsd' in df.columns:
        df['wsd'] = df['wsd'].replace([0, 99], np.nan)
        
    # PTY (강수형태): 0 ~ 7 제외 값 삭제
    if 'pty' in df.columns:
        # PTY는 0~7 사이의 정수 코드값으로 가정
        df.loc[(df['pty'] < 0) | (df['pty'] > 7), 'pty'] = np.nan

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