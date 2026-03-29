import pandas as pd
import numpy as np
import argparse
import os
import glob
import json
from datetime import datetime, timedelta
import logging
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
from variable_mapping import VALID_RANGE

# 경고 무시
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- 로깅 설정 ---
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # 핸들러 중복 방지
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger('MergeData')

# --- 데이터 소스별 컬럼 정리 규칙 정의 (요구사항 반영) ---

# 1. AIRKOREA: Merge 시 최종적으로 유지할 컬럼 목록
AIRKOREA_FINAL_COLS = [
    'geoId', 'geo_lon', 'geo_lat', 'dateTime', 'mangName', 'stationName', 
    'so2Value', 'coValue', 'o3Value', 'no2Value', 'pm10Value', 'pm25Value'
]

# 2. ODAM: Merge 시 최종적으로 제거할 컬럼 목록
ODAM_COLS_TO_DROP = ['st_lon', 'st_lat']

# 3. GEMS: Merge 시 최종적으로 제거할 컬럼 목록 (GEMS API 포맷 기반 메타데이터 제거)
GEMS_COLS_TO_DROP = [
    'st_lat', 'st_lon', 'sourceFile', 'datasetLevel', 'datasetCode', 'relativeAzimuthAngle', 
    'solarZenithAngle', 'viewingZenithAngle', 'solarAzimuthAngle', 'viewingAzimuthAngle', 
    'terrainHeight', 'groundPixelQualityFlags', 'land', 'cornerLatitude', 'cornerLongitude', 
    'aerosolEffectiveHeight', 'finalAlgorithmFlags', 'slantColumnAmountO2o2', 'surfaceAlbedo', 
    'aerosolType', 'normalizedRadiance', 'surfaceReflectance', 'uvAerosolIndex', 'visAerosolIndex',
    'backgroundReflectance', 'toaReflectance', 'wavelength', 'amfDiagnostic', 'gasProfile', 
    'layerField', 'scatteringWeight', 'terrainReflectivity', 'terrainPressure', 
    'algorithmQualityFlags', 'amfQualityFlags', 'aprioriTropNo2Profile', 'averagingKernel', 
    'reflectivity440', 'slantColumnAmountNo2', 'rootMeanSquareError', 'pressure', 'smape', 
    'importance', 'algorithmQualityFlag', 'amfTotal', 'reflectivity', 'slantColumnAmountSo2', 
    'dnaDamageIndex', 'plantResponseIndex', 'reflectanceAt354', 'surfacePhotolysisFrequencyO1D', 
    'tocQualityFlag', 'totalOzoneColumnUvi', 'uvIndex', 'vitaminDIndex', 'dataIndexFlag',
    # GEMS API 포맷에서 time은 dateTime으로 변환되었을 것으로 가정
    'time' 
]

# Merge Key (모든 소스 공통)을 key로 사용할 것]
MERGE_KEYS = ['geoId', 'geo_lon', 'geo_lat', 'dateTime', 'year', 'month', 'day']

def get_expected_periods(start_dt, end_dt, mode):
    """
    요청 기간의 기준 period 생성
    - hourly: 시간 단위 Timestamp
    - daily: 일 단위 date
    """
    if mode == "hourly":
        return pd.date_range(start=start_dt, end=end_dt + timedelta(days=1) - timedelta(hours=1), freq="h")
    elif mode == "daily":
        return pd.date_range(start=start_dt.date(), end=end_dt.date(), freq="d").date
    else:
        raise ValueError(f"지원하지 않는 mode: {mode}")
    
def load_existing_output_file(input_file, mode):
    """
    기존 merge 결과 파일(캐시용)을 읽는다.
    mode에 따라 hourly / daily 형식 처리.
    """
    if input_file is None or not os.path.exists(input_file):
        return None

    df = pd.read_csv(input_file)

    if mode == "hourly":
        if "dateTime" not in df.columns:
            raise ValueError(f"hourly 모드 input-file에는 'dateTime' 컬럼이 있어야 합니다: {input_file}")
        df["dateTime"] = pd.to_datetime(df["dateTime"], errors="coerce")
        df.dropna(subset=["dateTime"], inplace=True)

    elif mode == "daily":
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            df.dropna(subset=["date"], inplace=True)
        elif "dateTime" in df.columns:
            df["date"] = pd.to_datetime(df["dateTime"], errors="coerce").dt.date
            df.dropna(subset=["date"], inplace=True)
        else:
            raise ValueError(f"daily 모드 input-file에는 'date' 또는 'dateTime' 컬럼이 있어야 합니다: {input_file}")

    return df
    
def filter_existing_by_range(df, start_dt, end_dt, mode):
    """
    기존 캐시 파일에서 요청 기간만 남김
    """
    if df is None or df.empty:
        return None

    out = df.copy()

    if mode == "hourly":
        mask = (out["dateTime"] >= start_dt) & (out["dateTime"] <= (end_dt + timedelta(days=1) - timedelta(seconds=1)))
        out = out.loc[mask]
    else:
        mask = (out["date"] >= start_dt.date()) & (out["date"] <= end_dt.date())
        out = out.loc[mask]

    return out

def get_existing_periods(df, mode):
    """
    기존 결과가 이미 포함하는 period 집합 반환
    """
    if df is None or df.empty:
        return set()

    if mode == "hourly":
        return set(pd.to_datetime(df["dateTime"]).dt.floor("h"))
    else:
        return set(pd.to_datetime(df["date"]).dt.date)
    
def get_missing_periods(start_dt, end_dt, existing_df, mode):
    """
    요청 기간 중 기존 파일에 없는 period만 반환
    """
    expected = set(get_expected_periods(start_dt, end_dt, mode))
    existing = get_existing_periods(existing_df, mode)
    missing = sorted(expected - existing)
    return missing

def periods_to_dates(periods, mode):
    """
    missing period 목록을 파일 스캔용 date 집합으로 변환
    """
    if not periods:
        return set()
    if mode == "hourly":
        return set(pd.to_datetime(periods).date)
    else:
        return set(periods)
    
def build_master_df(source_dfs):
    """
    소스별 DataFrame을 outer join으로 병합
    """
    master_df = None

    for source, df in source_dfs.items():
        if master_df is None:
            master_df = df.copy()
        else:
            existing_cols = set(master_df.columns)
            new_cols = set(df.columns)
            overlap_cols = (existing_cols & new_cols) - set(MERGE_KEYS)

            if overlap_cols:
                logger.warning(f"  - {source}: 중복 컬럼 발견 (병합 시 제외됨): {overlap_cols}")
                df = df.drop(columns=list(overlap_cols))

            master_df = pd.merge(master_df, df, on=MERGE_KEYS, how='outer')

    if master_df is not None and not master_df.empty:
        master_df.sort_values(by=['geoId', 'dateTime'], inplace=True)

    return master_df

def make_daily_output(master_df):
    """
    master_df를 daily 결과로 변환
    """
    if master_df is None or master_df.empty:
        return pd.DataFrame()

    daily_df = master_df.copy()
    daily_df["date_key"] = daily_df["dateTime"].dt.date

    group_keys = ['date_key', 'geoId']

    data_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()
    data_cols = [c for c in data_cols if c not in group_keys]

    id_cols = [c for c in daily_df.columns if c not in data_cols and c not in group_keys]

    daily_mean_df = daily_df.groupby(group_keys)[data_cols].mean()
    daily_other_df = daily_df.groupby(group_keys)[id_cols].first()

    daily_final = daily_other_df.join(daily_mean_df, how='left').reset_index()
    daily_final["date"] = daily_final["date_key"]
    daily_final.drop(columns=["date_key", "dateTime", "hour"], errors="ignore", inplace=True)

    return daily_final

def get_period_series(df, mode):
    if mode == "hourly":
        return pd.to_datetime(df["dateTime"], errors="coerce").dt.floor("h")
    elif mode == "daily":
        if "date" in df.columns:
            return pd.to_datetime(df["date"], errors="coerce").dt.date
        return pd.to_datetime(df["dateTime"], errors="coerce").dt.date
    else:
        raise ValueError(f"지원하지 않는 mode: {mode}")

    
def get_source_presence_periods(df, source, mode):
    """
    merge된 최종 df에서 source 값이 실제 존재하는 period 집합 반환
    """
    if df is None or df.empty:
        return set()

    if source == "AIRKOREA":
        value_cols = [
        'stationName', 'mangName',
        'so2Value', 'coValue', 'o3Value', 'no2Value', 'pm10Value', 'pm25Value'
        ]
    else:
        valid_range = VALID_RANGE.get(source, {})
        base_cols = list(valid_range.keys())

        value_cols = []
        for base_col in base_cols:
            if base_col in df.columns:
                value_cols.append(base_col)
            value_cols.extend([c for c in df.columns if c.startswith(f"{base_col}_")])

        value_cols = list(dict.fromkeys(value_cols))

    if not value_cols:
        logger.warning(f"⚠️ {source}: 존재 여부를 판단할 컬럼이 없습니다.")
        return set()

    mask = df[value_cols].notna().any(axis=1)
    period_series = get_period_series(df.loc[mask].copy(), mode)
    return set(period_series.dropna())

def get_common_periods(df, mode, sources):
    """
    지정한 sources의 공통 period 반환
    """
    if df is None or df.empty:
        return set()

    period_sets = []
    for source in sources:
        pset = get_source_presence_periods(df, source, mode)
        logger.info(f"  → {source} 값 존재 period 수: {len(pset)}")
        period_sets.append(pset)

    if not period_sets:
        return set()

    common_periods = set.intersection(*period_sets)
    logger.info(f"  → 공통 period 수: {len(common_periods)}")
    return common_periods


def filter_by_periods(df, periods, mode):
    if df is None or df.empty:
        return df.copy()

    if not periods:
        return df.iloc[0:0].copy()

    out = df.copy()
    period_series = get_period_series(out, mode)
    return out.loc[period_series.isin(periods)].copy()


def filter_airkorea_rows(df):
    if df is None or df.empty:
        return df.copy()

    air_cols = [
        'stationName', 'mangName',
        'so2Value', 'coValue', 'o3Value', 'no2Value', 'pm10Value', 'pm25Value'
        ]
    if not air_cols:
        logger.warning("⚠️ AIRKOREA 값 컬럼이 없습니다.")
        return df.iloc[0:0].copy()

    mask = df[air_cols].notna().any(axis=1)
    return df.loc[mask].copy()

def one_hot_encode(df, data_source, exclude_keys=None):
    """
    valid range에 정의된 범주형 변수만 one-hot encoding 수행
    """
    if exclude_keys is None:
        exclude_keys = []
    
    valid_range = VALID_RANGE.get(data_source, {})
    cat_cols = [
        col for col, meta in valid_range.items()
        if meta.get("type") == "cat" and col in df.columns and col not in exclude_keys
    ]

    if not cat_cols:
        return df
    
    for col in cat_cols:
        valid_values = list(valid_range[col]["values"])

        # 유효값 아니면 99 처리
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[~df[col].isin(valid_values), col] = 99

        if 99 not in valid_values:
            valid_values.append(99)

        # 각 valid range 값에 대해 dummy 컬럼 생성
        for val in valid_values:
            dummy_col = f"{col}_{val}"
            df[dummy_col] = (df[col] == val).astype(np.uint8)

        # 원본 범주형 컬럼 제거
        df.drop(columns=[col], inplace=True)

    return df

# --- 병렬 파일 로드 및 전처리 함수 ---
def load_and_preprocess_file(filepath, source):
    """CSV 파일 1개를 로드하고 소스별 전처리를 수행합니다."""
    try:
        df = pd.read_csv(filepath)

        if source == 'AIRKOREA':
            # AIRKOREA: 명시된 12개 열만 선택
            cols_to_keep = [c for c in AIRKOREA_FINAL_COLS if c in df.columns]
            df = df[cols_to_keep]
        elif source == 'ODAM':
            # ODAM: st_lon, st_lat 열 drop
            df.drop(columns=ODAM_COLS_TO_DROP, errors='ignore', inplace=True)
        elif source == 'GEMS':
            # GEMS: 메타데이터 열 drop
            df.drop(columns=GEMS_COLS_TO_DROP, errors='ignore', inplace=True)

        # 원핫인코딩
        df = one_hot_encode(df, data_source= source, exclude_keys= MERGE_KEYS)

        return df
    except Exception as e:
        logger.warning(f"  - {source} 파일 로드/정리 오류 ({os.path.basename(filepath)}): {e}")
        return None

def load_source_data_parallel(input_dir, source, start_dt, end_dt, source_dirs=None, target_dates=None, target_periods=None, mode="hourly"):
    """지정된 기간과 소스에 대해 시간별 CSV를 병렬로 로드하고 컬럼을 정리합니다.
    
    Args:
        input_dir: 기본 입력 디렉토리 (source_dirs가 없을 때 사용)
        source: 데이터 소스명 (GK2A, ODAM, AIRKOREA, GEMS)
        start_dt: 시작 날짜
        end_dt: 종료 날짜
        source_dirs: 소스별 경로 매핑 dict (optional, 예: {'GK2A': '/app/data/GK2A/2025/10/'})
    """
    
    # 1. 입력 폴더 경로 설정
    if source_dirs and source in source_dirs:
        # JSON으로 지정된 경로가 있으면 해당 경로 사용
        source_dir = source_dirs[source]
        logger.info(f"  → {source}: 사용자 지정 경로 사용 ({source_dir})")
    else:
        # 기존 방식: input_dir + 고정 폴더명
        if source == 'GK2A': output_name = "KOMPSAT_LE2"
        elif source == 'ODAM': output_name = "WEATHER_ODAM"
        elif source == 'AIRKOREA': output_name = "AIRKOREA"
        elif source == 'GEMS': output_name = "GEMS"
        else: return None
            
        source_dir = os.path.join(input_dir, output_name)
        
    if not os.path.isdir(source_dir):
        logger.warning(f"⚠️ {source}: 입력 디렉토리를 찾을 수 없습니다: {source_dir}")
        return None
    
    all_source_files = []
    if target_dates is None:
        target_dates = set()
        current_dt = start_dt
        while current_dt <= end_dt:
            target_dates.add(current_dt.date())
            current_dt += timedelta(days=1)
    
    # 2. 기간 내 파일 스캔
    logger.info(f"  → {source}: 파일 스캔 중... ({min(target_dates)} ~ {max(target_dates)})")

    # 일(day) 폴더 기준으로 스캔하여 하위 시간별 CSV 모두 찾기
    for d in sorted(target_dates):
        search_path = os.path.join(source_dir, str(d.year), str(d.month), str(d.day), "*.csv")
        all_source_files.extend(glob.glob(search_path, recursive=False))
        
    if not all_source_files:
        logger.warning(f"⚠️ {source}: 지정된 기간 내에 처리할 CSV 파일을 찾지 못했습니다.")
        return None
    
    logger.info(f"  → {source}: 총 {len(all_source_files)}개 파일 병렬 로드 시작.")
    
    # 3. 병렬 처리
    pool = Pool(processes=max(1, cpu_count() // 2))
    # partial을 사용하여 load_and_preprocess_file 함수에 'source' 인자를 고정
    load_func = partial(load_and_preprocess_file, source=source)
    
    df_list = pool.map(load_func, all_source_files)
    
    pool.close()
    pool.join()
    
    # None이 반환된 (실패한) 케이스 제거
    df_list = [df for df in df_list if df is not None and not df.empty]
    
    if not df_list: 
        logger.error(f"❌ {source}: 모든 파일 로드/처리에 실패했습니다.")
        return None
    
    # 4. 데이터 합치기 및 Key 포맷 통일
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df['dateTime'] = pd.to_datetime(combined_df['dateTime'],errors='coerce')
    
    if target_periods is not None:
        if mode == "hourly":
            target_periods = set(pd.to_datetime(list(target_periods)))
            combined_df = combined_df[combined_df['dateTime'].dt.floor("h").isin(target_periods)].copy()
        elif mode == "daily":
            target_periods = set(target_periods)
            combined_df = combined_df[combined_df['dateTime'].dt.date.isin(target_periods)].copy()
        
    logger.info(f"  → {source}: 최종 {len(combined_df)}개 레코드 준비 완료.")
    return combined_df

# --- 메인 실행 함수 ---
def main_merge():
    parser = argparse.ArgumentParser(description="[프로세스 4] 모든 데이터 소스의 결측치 보간 완료 파일을 합치고, 일자별/시간별로 출력합니다.")
    
    # 1. 입력 파라미터 (명시된 요구사항 반영)
    parser.add_argument('--mode', type=str, required=True, choices=['hourly', 'daily'], 
                        help='출력 모드 (hourly: 모든 시간 0-23 / daily: 하루 평균값)')
    parser.add_argument('--start-date', type=str, required=True, help='시작 일자 (YYYYMMDD 형식, 예: 20230809)')
    parser.add_argument('--end-date', type=str, required=True, help='종료 일자 (YYYYMMDD 형식, 예: 20250505)')
    parser.add_argument('--sources', nargs='+', required=True, 
                        choices=['GK2A', 'ODAM', 'GEMS', 'AIRKOREA'],
                        help='합칠 데이터 소스 선택 (복수 선택 가능, 예: GK2A AIRKOREA)')
    parser.add_argument('--input-dir', '-i', type=str, required=False, 
                        help='입력 데이터(프로세스 3의 출력 경로)의 *최상위* 폴더 경로 (예: HOURLY_DATA_IMPUTED). --source-dirs가 없을 때 필수')
    parser.add_argument('--output-dir', '-o', type=str, required=True, 
                        help='최종 산출물(FIN 폴더)을 저장할 *최상위* 폴더 경로 (예: FINAL_OUTPUT)')
    parser.add_argument('--source-dirs', type=str, required=False,
                        help='소스별 경로 매핑 (JSON 형식, 예: {"GK2A":"/app/data/GK2A/2025/10/","AIRKOREA":"/app/data/AIRKOREA_added/2025/10"})')
    parser.add_argument('--input-file', type=str, required=False, default=None,
                        help='기존 merge 결과 캐시 파일(csv). 있으면 요청 기간 중 없는 기간만 추가 로드 및 제외 기간 삭제')
    parser.add_argument(
    '--save-types',
    nargs='+',
    required=True,
    choices=['outer_all', 'common_period', 'common_period_airkorea_only'],
    help=(
        "저장 방식 선택 (복수 선택 가능)\n"
        "- outer_all: 전체 데이터 outer merge 저장\n"
        "- common_period: ODAM/GK2A/AIRKOREA 공통 period만 저장\n"
        "- common_period_airkorea_only: 공통 period 중 AIRKOREA 값이 있는 행만 저장"
    ))

    args = parser.parse_args()
    
    # source-dirs JSON 파싱
    source_dirs_dict = None
    if args.source_dirs:
        try:
            source_dirs_dict = json.loads(args.source_dirs)
            logger.info(f"  → 소스별 경로 매핑 사용: {source_dirs_dict}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ --source-dirs JSON 파싱 오류: {e}")
            return
    
    # input-dir과 source-dirs 중 하나는 필수
    if not args.input_dir and not source_dirs_dict:
        logger.error("❌ --input-dir 또는 --source-dirs 중 하나는 반드시 지정해야 합니다.")
        return
    
    logger.info("="*80)
    logger.info(f"🔗 [프로세스 4: 데이터 합치기] 시작 (모드: {args.mode.upper()})")
    
    try:
        # YYYYMMDD 형식의 날짜를 datetime 객체로 변환
        start_dt = datetime.strptime(args.start_date, '%Y%m%d')
        end_dt = datetime.strptime(args.end_date, '%Y%m%d')
    except ValueError:
        logger.error("❌ 날짜 형식 오류: 시작일 또는 종료일을 YYYYMMDD 형식으로 입력하세요.")
        return
    
    # 1. 기존 캐시 파일 로드
    existing_df = load_existing_output_file(args.input_file, args.mode)
    existing_df = filter_existing_by_range(existing_df, start_dt, end_dt, args.mode)

    if existing_df is not None:
        logger.info(f"  → 기존 input-file 로드 완료: {len(existing_df)}행")
    else:
        logger.info("  → 기존 input-file 없음. 전체 기간 새로 merge 수행")

    # 2. 없는 기간만 계산
    missing_periods = get_missing_periods(start_dt, end_dt, existing_df, args.mode)

    if len(missing_periods) == 0:
        logger.info("  → 요청 기간이 모두 기존 input-file에 존재합니다. 신규 로드 없이 재사용합니다.")
        newly_built_output = pd.DataFrame()
    else:
        logger.info(f"  → 기존 파일에 없는 period 수: {len(missing_periods)}")

        target_dates = periods_to_dates(missing_periods, args.mode)

        # 3. 필요한 기간만 소스별 로드
        source_dfs = {}
        for source in args.sources:
            df = load_source_data_parallel(
                args.input_dir if args.input_dir else "",
                source,
                start_dt,
                end_dt,
                source_dirs_dict,
                target_dates=target_dates,
                target_periods=missing_periods,
                mode=args.mode
            )
            if df is not None and not df.empty:
                source_dfs[source] = df

        if not source_dfs:
            logger.warning("⚠️ 신규로 추가할 유효한 소스 데이터가 없습니다.")
            newly_built_output = pd.DataFrame()
        else:
            logger.info("  → 모든 소스 데이터 Outer Join 시작...")
            master_df = build_master_df(source_dfs)

            if master_df is None or master_df.empty:
                newly_built_output = pd.DataFrame()
            else:
                logger.info(f"  → 신규 Master DataFrame 생성 완료. (총 {len(master_df)}행, {len(master_df.columns)}열)")

                if args.mode == 'hourly':
                    newly_built_output = master_df.copy()
                elif args.mode == 'daily':
                    newly_built_output = make_daily_output(master_df)
                else:
                    raise ValueError(f"지원하지 않는 mode: {args.mode}")

    # 4. 기존 결과 + 신규 결과 합치기
    if existing_df is not None and not existing_df.empty:
        if newly_built_output is not None and not newly_built_output.empty:
            final_output_df = pd.concat([existing_df, newly_built_output], ignore_index=True)
        else:
            final_output_df = existing_df.copy()
    else:
        final_output_df = newly_built_output.copy() if newly_built_output is not None else pd.DataFrame()

    if final_output_df is None or final_output_df.empty:
        logger.error("❌ 최종 결과가 비어 있습니다.")
        return
            
    # 5. 중복 제거 + 정렬
    if args.mode == "hourly":
        final_output_df["dateTime"] = pd.to_datetime(final_output_df["dateTime"], errors="coerce")
        final_output_df.dropna(subset=["dateTime"], inplace=True)
        final_output_df.drop_duplicates(subset=MERGE_KEYS, keep="last", inplace=True)
        final_output_df.sort_values(by=["geoId", "dateTime"], inplace=True)

    elif args.mode == "daily":
        final_output_df["date"] = pd.to_datetime(final_output_df["date"], errors="coerce").dt.date
        final_output_df.dropna(subset=["date"], inplace=True)

        dedup_keys = ["date", "geoId"]
        if "geo_lon" in final_output_df.columns:
            dedup_keys.append("geo_lon")
        if "geo_lat" in final_output_df.columns:
            dedup_keys.append("geo_lat")

        final_output_df.drop_duplicates(subset=dedup_keys, keep="last", inplace=True)
        final_output_df.sort_values(by=["geoId", "date"], inplace=True)

    logger.info(f"  → 최종 결과 준비 완료. (총 {len(final_output_df)}행, {len(final_output_df.columns)}열)")

    # 6. input-file 업데이트(캐시 파일로 재사용)
    # input file의 name을 변경할 필요가 있음. 규칙 적용 필요.
    if args.input_file:
        os.makedirs(os.path.dirname(args.input_file), exist_ok=True)
        save_df = final_output_df.copy()

        if args.mode == "hourly" and "dateTime" in save_df.columns:
            save_df["dateTime"] = pd.to_datetime(save_df["dateTime"]).dt.strftime('%Y-%m-%d %H:%M:%S')
        elif args.mode == "daily" and "date" in save_df.columns:
            save_df["date"] = pd.to_datetime(save_df["date"]).dt.strftime('%Y-%m-%d')

        save_df.to_csv(args.input_file, index=False)
        logger.info(f"  → input-file 업데이트 완료: {args.input_file}")
    

    # 7. 저장용 DataFrame 생성
    save_targets = {}

    # 1) 전체 outer merge
    if "outer_all" in args.save_types:
        save_targets["outer_all"] = final_output_df.copy()

    # 공통 기준 source
    common_sources = [s for s in ["ODAM", "GK2A", "AIRKOREA"] if s in args.sources]

    if ("common_period" in args.save_types) or ("common_period_airkorea_only" in args.save_types):
        common_periods = get_common_periods(final_output_df, args.mode, common_sources)
        common_df = filter_by_periods(final_output_df, common_periods, args.mode)

        if "common_period" in args.save_types:
            save_targets["common_period"] = common_df.copy()

        if "common_period_airkorea_only" in args.save_types:
            save_targets["common_period_airkorea_only"] = filter_airkorea_rows(common_df)

    # 8. 파일 저장
    os.makedirs(args.output_dir, exist_ok=True)

    for save_type, save_df in save_targets.items():
        if save_df is None or save_df.empty:
            logger.warning(f"⚠️ {save_type}: 저장할 데이터가 비어 있습니다.")
            continue

        output_filename = f"FIN_{save_type}_{args.mode}_{args.start_date}_{args.end_date}.csv"
        output_path = os.path.join(args.output_dir, output_filename)

        save_out = save_df.copy()

        if args.mode == "hourly" and "dateTime" in save_out.columns:
            save_out["dateTime"] = pd.to_datetime(save_out["dateTime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

        if args.mode == "daily":
            if "date" in save_out.columns:
                save_out["date"] = pd.to_datetime(save_out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            elif "dateTime" in save_out.columns:
                save_out["dateTime"] = pd.to_datetime(save_out["dateTime"], errors="coerce").dt.strftime("%Y-%m-%d")

        save_out.to_csv(output_path, index=False)
        logger.info(f"  → 저장 완료: {output_path} ({len(save_out)}행)")
    
        
    logger.info("="*80)
    logger.info("✅ [프로세스 4: 데이터 합치기] 작업 완료.")
    logger.info("="*80)

if __name__ == "__main__":
    main_merge()