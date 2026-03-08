import pandas as pd
import numpy as np
import joblib
import argparse
import os
import glob
import logging
from sklearn.experimental import enable_iterative_imputer # IterativeImputer를 사용하기 위해 필요
from sklearn.impute import IterativeImputer
from datetime import datetime, timedelta

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

def clean_and_prepare_data(df, data_source):
    """
    DQF/Flag 컬럼을 제거하고, Imputation에 필요한 데이터만 준비합니다.
    """
    if df.empty:
        logger.warning("입력 DataFrame이 비어 있습니다. 처리할 데이터가 없습니다.")
        return None, None
        
    # --- [필수 요구사항] GK2A의 경우 'dqf' 또는 'flag'가 포함된 열은 모두 drop 시킵니다. ---
    # ODAM은 DQF/Flag가 없으나, 안전하게 모든 데이터 소스에 대해 DQF/flag 제거 로직을 적용합니다.
    
    # 1. DQF/Flag 컬럼 제거
    columns_to_drop = [col for col in df.columns if ('dqf' in col.lower() or 'flag' in col.lower())]
    if columns_to_drop:
        df_cleaned = df.drop(columns_to_drop, axis=1, errors='ignore')
        logger.info(f"  → DQF/Flag 컬럼 {len(columns_to_drop)}개 제거 완료: {columns_to_drop}")
    else:
        df_cleaned = df.copy()

    # 2. Imputation 대상에서 제외할 컬럼(Geo/Time ID) 분리
    id_cols = ['geoId', 'dateTime', 'geo_lon', 'geo_lat']
    cols_to_drop = [col for col in id_cols if col in df_cleaned.columns]
    data_df = df_cleaned.drop(cols_to_drop, axis=1, errors='ignore')
    
    # 3. 모든 데이터 컬럼을 숫자형으로 변환 (Imputer 요구사항)
    for col in data_df.columns:
        if not pd.api.types.is_numeric_dtype(data_df[col]):
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    
    # NaN만 남은 컬럼 제거 (IterativeImputer 오류 방지)
    data_df.dropna(axis=1, how='all', inplace=True)
    
    logger.info(f"  → 최종 Imputation 대상 컬럼 수: {len(data_df.columns)}")
    
    # DQF/Flag가 제거된 DataFrame에서 ID 컬럼만 추출
    id_df = df_cleaned[['geoId', 'dateTime']].copy()
    
    return data_df, id_df



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

def main_train():
    parser = argparse.ArgumentParser(description="[프로세스 3] IterativeImputer 모델을 학습하고 저장합니다.")
    
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
    
    args = parser.parse_args()
    
    # 1. 데이터 로드 및 전처리
    logger.info("="*60)
    logger.info(f"🧠 [프로세스 3: Imputer 훈련] 시작 ({args.data_source})")
    
    # 날짜 범위 계산 (pivot-date 기반 롤링윈도우)
    start_date_dt = None
    end_date_dt = None
    
    if args.pivot_date:
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

    data_to_impute, _ = clean_and_prepare_data(raw_df, args.data_source)
    
    if data_to_impute is None or data_to_impute.empty:
        logger.error("❌ Imputation 대상 데이터셋이 비어 있거나 유효하지 않습니다. 스크립트를 종료합니다.")
        return
        
    # 2. IterativeImputer 훈련
    logger.info("  → IterativeImputer 모델 훈련 시작...")
    
    try:
        imputer = IterativeImputer(random_state=42, max_iter=10)
        imputer.fit(data_to_impute)
        
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