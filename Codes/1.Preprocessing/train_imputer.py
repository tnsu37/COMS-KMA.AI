import os
import glob
import joblib
import argparse
import logging
import warnings
import numpy as np
import pandas as pd

from datetime import timedelta
from lightgbm import LGBMClassifier, LGBMRegressor
from missforest import MissForest
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# 로깅 설정 (process_outliers.py와 동일)
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger('TrainImputer')

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

# GK2A의 연속형/범주형 변수의 유효범위
# 모델링 제외 변수는 주석처리
VALID_RANGE_GK2A = {
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
        'st_lon': {'type': 'con', 'values': [-np.inf, np.inf]},
        'st_lat': {'type': 'con', 'values': [-np.inf, np.inf]},
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

categorical_cols = [
    col for col, info in VALID_RANGE_GK2A.items() if info["type"] == "cat"
]

continuous_cols = [
    col for col, info in VALID_RANGE_GK2A.items() if info["type"] == "con"
]

# continuous_cols = [
#     'ctps_cth', 'ctps_ctp', 'ctps_ctt', 'cla_cloud_fraction',
#     'dcoew_radius', 'dcoew_thickness', 'dcoew_liquid_path', 'ncot',
#     'ci_ci2', 'rr', 'qpn_rate', 'tpw_low', 'tpw_mid',
#     'tpw_high', 'tpw', 'aii_tti', 'swrad_downward', 'swrad_absorbed',
#     'rr_raining_ct_flag', 'st_lon', 'st_lat'          
#                    ]
# categorical_cols = [
#     'cld', 'ctps_cp', 'cla_type', 'ci_ci1_ccm', 'ctps_dqf1',
#     'cla_type_dqf', 'cla_cloud_fraction_dqf', 'dcoew_dqf1', 'ncot_dqf',
#     'qpn_dqf1', 'tpw_dqf1', 'tpw_dqf2', 'aii_dqf1', 'aii_dqf2',
#     'swrad_downward_dqf', 'swrad_absorbed_dqf', 'swrad_dqf1'
# ]

def apply_categorical_outlier_to_99(df):
    """범주형 변수에서 허용된 값 외의 값은 99로 치환"""
    
    for col, meta in VALID_RANGE_GK2A.items():
        if meta.get("type") == "cat" and col in df.columns:
            valid_values = set(meta["values"])
            
            # NaN 제외하고 invalid만 처리
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


def clean_and_prepare_data(df, data_source):
    if df.empty:
        logger.warning("입력 DataFrame이 비어 있습니다. 처리할 데이터가 없습니다.")
        return None, None, None
    
    # 1. NaN만 남은 컬럼 제거 (오류 방지)
    df.dropna(axis=1, how='all', inplace=True)
    
    columns_to_drop = [
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
    if data_source == "GK2A":
        df_cleaned = df.drop(columns_to_drop, axis=1, errors='ignore')
        logger.info(f"  → GK2A - 보간 제외 대상 컬럼 {len(columns_to_drop)}개 제거 완료: {columns_to_drop}")

        # 2. 규칙 기반 보간 수행
        df_cleaned = rule_based_imputation(df_cleaned)
        logger.info(f"  → 규칙 기반 보간 완료")

    else: df_cleaned = df.copy()

    # 3. Imputation 대상에서 제외할 컬럼(Geo/Time ID) 분리
    id_cols = ['geoId', 'dateTime', 'geo_lon', 'geo_lat']
    cols_to_drop = [col for col in id_cols if col in df_cleaned.columns]
    data_df = df_cleaned.drop(cols_to_drop, axis=1, errors='ignore')

    # 4. 모든 데이터 컬럼을 숫자형으로 변환 (Imputer 요구사항)
    for col in data_df.columns:
        if not pd.api.types.is_numeric_dtype(data_df[col]):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    # 5. cat_cols
    final_cols = data_df.columns.tolist()
    cat_cols = [col for col in categorical_cols if col in data_df.columns]

    logger.info(f"  → 최종 Imputation 대상 컬럼 수: {len(final_cols)}")
    logger.info(f"  → 최종 Imputation 대상 행 수: {len(data_df)}")
    
    # DQF/Flag가 제거된 DataFrame에서 ID 컬럼만 추출
    id_df = df_cleaned[['geoId', 'dateTime']].copy()
    
    return data_df, id_df, cat_cols


def load_data_for_training(input_dir, data_source, start_date=None, end_date=None):
    """입력 폴더에서 CSV를 로드하고 병합합니다."""
    
    # --- [수정] 데이터 소스 인자를 실제 디렉토리명으로 매핑 ---
    if data_source == 'GK2A':
        source_dir_name = "KOMPSAT_LE2"
    elif data_source == 'ODAM':
        source_dir_name = "WEATHER_ODAM"
    elif data_source == 'GEMS':
        source_dir_name = "GEMS"
    else:
        source_dir_name = data_source

    # [수정] input_dir 바로 하위가 아닌, 매핑된 폴더명을 기준으로 검색
    input_path = os.path.join(input_dir, source_dir_name)
    logger.info(f"  → 훈련 데이터 로드 시작: {input_path}")
    
    # [수정] 검색 경로 변경
    all_files = glob.glob(os.path.join(input_path, "**", "*.csv"), recursive=True)
    
    if not all_files:
        logger.error(f"❌ '{input_path}'에서 CSV 파일을 찾을 수 없습니다.")
        return None
    
    df_list = []
    
    for filepath in all_files:
        try:
            df = pd.read_csv(filepath)
            # 날짜 필터링 로직 추가 (필요하다면)
            if start_date and end_date:
                if 'dateTime' in df.columns:
                    df['dateTime'] = pd.to_datetime(df['dateTime'], errors='coerce')
                    df = df[(df['dateTime'] >= start_date) & (df['dateTime'] <= end_date)]
                
            df_list.append(df)
        except Exception as e:
            logger.warning(f"  - 파일 로드 오류 ({os.path.basename(filepath)}): {e}")
            
    if not df_list:
        logger.error("❌ 필터링 조건을 만족하는 데이터가 없습니다.")
        return None
        
    # 모든 데이터를 합칠 때, 겹치는 geoId+dateTime을 처리하기 위해 중복 제거 로직 필요
    combined_df = pd.concat(df_list, ignore_index=True)
    
    if 'geoId' in combined_df.columns and 'dateTime' in combined_df.columns:
        # geoId와 dateTime을 기준으로 중복 제거 (가장 마지막 값 유지)
        combined_df.sort_values(by='dateTime', inplace=True)
        combined_df.drop_duplicates(subset=['geoId', 'dateTime'], keep='last', inplace=True)
        
    logger.info(f"  → 총 {len(combined_df)}개의 레코드 로드 및 병합 완료.")
    return combined_df


def train_missforest(data_df, cat_cols, random_state=42):
    logger.info("  → MissForest 모델 훈련 시작...")

    clf = LGBMClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1
    )

    rgr = LGBMRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1
    )

    imputer = MissForest(
        clf=clf,
        rgr=rgr,
        categorical=cat_cols if cat_cols else None,
        initial_guess="median",
        max_iter=3,
        early_stopping=True,
        verbose=2
    )

    imputer.fit(data_df)
    return imputer


def main_train():
    parser = argparse.ArgumentParser(description="[프로세스 3] MissForest 모델을 학습하고 저장합니다.")
    
    parser.add_argument('--data-source', '-s', type=str, required=True, 
                        choices=['GK2A', 'ODAM'],
                        help='Imputer를 학습할 데이터 소스 (GK2A 또는 ODAM). AIRKOREA는 제외.')
    parser.add_argument('--input-dir', '-i', type=str, required=True, 
                        help='훈련 데이터(클리닝 완료된 시간별 CSV 폴더) 경로. (예: HOURLY_DATA/KOMPSAT_LE2)')
    parser.add_argument('--output-path', '-o', type=str, required=True, 
                        help='학습된 Imputer 모델(.pkl)을 저장할 파일 경로. {pivot} 플레이스홀더 사용 가능 (예: imputers/gk2a_{pivot}.pkl)')
    
    # [롤링윈도우 모드] 기준 날짜 지정 (한달 전부터 학습)
    parser.add_argument('--pivot-date', type=str, default=None,
                        help='기준 날짜 (YYYY-MM-DD). 이 날짜로부터 한달 전까지의 데이터로 학습. 미지정 시 전체 데이터 사용.')
    parser.add_argument('--training-days', type=int, default=30,
                        help='학습 데이터 기간 (일). 기본값: 30일 (한달). pivot-date로부터 이 기간만큼 과거 데이터 사용.')
    
    # 과거 데이터 처리를 위한 '전체 기간 학습' 옵션 (추가 요구사항 반영)
    parser.add_argument('--past-mode', action='store_true', 
                        help='과거 데이터 전체 기간 학습 모드 활성화 (Imputer를 별도 저장해야 함).')
    
    # 학습 기간 시작 날짜, 종료 날짜 직접 지정
    parser.add_argument('--start-date', type=str, default=None, 
                        help='학습 시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='학습 종료 날짜 (YYYY-MM-DD)')
    

    args = parser.parse_args()
    
    # 1. 데이터 로드 및 전처리
    logger.info("="*60)
    logger.info(f"🧠 [프로세스 3: Imputer 훈련] 시작 ({args.data_source})")
    
    # 날짜 범위 계산 (pivot-date 기반 롤링윈도우)
    start_date_dt = None
    end_date_dt = None

    # 1순위: start-date / end-date 직접 지정
    if args.start_date and args.end_date:
        try:
            start_date_dt = pd.to_datetime(args.start_date)
            end_date_dt = pd.to_datetime(args.end_date) + timedelta(days=1) - timedelta(seconds=1)

            logger.info(f"  → 직접 지정 학습 기간: {args.start_date} ~ {args.end_date}")
        except Exception:
            logger.error("❌ start-date / end-date 형식이 잘못되었습니다. YYYY-MM-DD 형식이어야 합니다.")
            return
    
    # 2순위: pivot-date 기반
    elif args.pivot_date:
        try:
            pivot_dt = pd.to_datetime(args.pivot_date)
            end_date_dt = pivot_dt
            start_date_dt = pivot_dt - timedelta(days=args.training_days)
            
            logger.info(f"  → 기준 날짜 (Pivot): {args.pivot_date}")
            logger.info(f"  → 학습 기간: {args.training_days}일 ({start_date_dt.strftime('%Y-%m-%d')} ~ {end_date_dt.strftime('%Y-%m-%d')})")
            
        except Exception as e:
            logger.error(f"❌ 잘못된 pivot-date 형식: {args.pivot_date}. YYYY-MM-DD 형식이어야 합니다.")
            return
    
    raw_df = load_data_for_training(args.input_dir, args.data_source, start_date_dt, end_date_dt)
    if raw_df is None:
        logger.error("❌ 훈련을 위한 데이터 로드 실패. 스크립트를 종료합니다.")
        return
    
    # 로드된 데이터 개수 확인
    if start_date_dt and end_date_dt:
        logger.info(f"  → 로드된 레코드 중 날짜 범위 내: {len(raw_df)}개")

    data_to_impute, _, cat_cols = clean_and_prepare_data(raw_df, args.data_source)
    
    if data_to_impute is None or data_to_impute.empty:
        logger.error("❌ Imputation 대상 데이터셋이 비어 있거나 유효하지 않습니다. 스크립트를 종료합니다.")
        return
    
    if len(data_to_impute) > 5000000:
        logger.info(f"  → 샘플링 전 행 수: {len(data_to_impute)}")
        data_to_impute = data_to_impute.sample(n=5000000, random_state=42).reset_index(drop=True)
        logger.info(f"  → 샘플링 후 행 수: {len(data_to_impute)}")
        
    # 2. MissForest 훈련
    logger.info("  → MissForest 모델 훈련 시작...")
    
    try:
        imputer = train_missforest(
            data_df=data_to_impute,
            cat_cols=cat_cols,
            random_state=42
        )
        
        # 3. 모델 저장
        output_path = args.output_path
        
        # {pivot} 플레이스홀더 자동 치환
        if args.pivot_date and '{pivot}' in output_path:
            output_path = output_path.replace('{pivot}', args.pivot_date)
            logger.info(f"  → 파일명 자동 생성: pivot={args.pivot_date}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(imputer, output_path)
        logger.info(f"✅ Imputer 훈련 및 저장 완료: {output_path}")
        
        if args.pivot_date:
            logger.info(f"  → 학습 기준: Pivot={args.pivot_date}, Training={args.training_days}일")

    except Exception as e:
        logger.exception(f"❌ Imputer 훈련 중 치명적인 오류 발생: {e}")
        
    logger.info("="*60)

if __name__ == "__main__":
    main_train()