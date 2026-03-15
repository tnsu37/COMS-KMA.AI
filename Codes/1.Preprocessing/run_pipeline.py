import sys
import argparse
import subprocess
import logging
from datetime import datetime
import importlib.util

# --- 로깅 설정 ---
def setup_logger():
    """총괄 파이프라인 로거를 설정합니다."""
    logger = logging.getLogger('PipelineManager')
    logger.setLevel(logging.INFO)
    
    # 핸들러 중복 방지
    if logger.handlers:
        return logger
        
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

# --- 프로세스별 실행 스크립트 매핑 ---
# p1and2는 process_outliers.py를 사용합니다.
SCRIPT_MAP = {
    'p1': 'create_coordinate_mapping.py',
    'p2': 'process_outliers_v3_2.py',
    'p3_train': 'train_imputer.py',
    'p3_impute': 'impute_hourly.py',
    'p4': 'merge_data.py',
    'p1and2': 'process_outliers.py'
}

def main():
    """
    프로세스 총괄 인터페이스 메인 함수
    """
    
    # 1. 로거 설정
    logger = setup_logger()

    # 2. 메인 파서 설정
    parser = argparse.ArgumentParser(
        description="데이터 클리닝 파이프라인 총괄 스크립트",
        epilog="""
--- [사용 예시] ---

  # [P1] GK2A용 좌표 매칭 테이블 생성 (최초 1회)
  python run_pipeline.py p1 --geoid /app/data/1km_grid_ko_full.csv --latlon /app/data/gk2a_ami_ko020lc_latlon.nc --output /app/data/matched_geoid.csv

  # [P1&2] GK2A 과거 데이터(.nc) 전체 처리
  python run_pipeline.py p1and2 --input-sample /app/data/GK2A_KMAHDD --input-match-table /app/data/matched_geoid.csv --output /app/data/HOURLY_DATA --resume

  # [P2] API 데이터 이상치 처리
  python run_pipeline.py p2 --data-type AIRKOREA_API --input-dir /app/data/API_AIRKOREA_raw --output /app/data/HOURLY_DATA

  # [P3-Train] Imputer 훈련 (예: 2023년 1월 데이터로 '2023-01' 월간 모델 훈련)
  python run_pipeline.py p3_train --data-source GK2A --input-dir /app/data/HOURLY_DATA --output-path /app/data/imputers/gk2a_imputer_2023-01.pkl --start-date 2023-01-01 --end-date 2023-01-31

  # [P3-Impute] 결측치 보간 (훈련된 'month' 단위 Imputer 사용)
  python run_pipeline.py p3_impute --data-source GK2A --input /app/data/HOURLY_DATA --input_imputer /app/data/imputers --output /app/data/HOURLY_DATA_IMPUTED --period month --resume

  # [P4] 최종 병합 (일별 평균)
  python run_pipeline.py p4 --mode daily --start-date 20230101 --end-date 20241231 --sources GK2A AIRKOREA --input-dir /app/data/HOURLY_DATA_IMPUTED --output-dir /app/data/FINAL_OUTPUT
---------------------
""",
        formatter_class=argparse.RawTextHelpFormatter # 예시가 잘 보이도록
    )
    
    parser.add_argument(
        'process',
        choices=SCRIPT_MAP.keys(),
        help=f"실행할 프로세스 ID: {list(SCRIPT_MAP.keys())}"
    )
    
    # 3. 인자 파싱
    args, unknown_args = parser.parse_known_args()
    
    # 4. 실행할 스크립트 결정
    script_to_run = SCRIPT_MAP[args.process]
    
    logger.info("="*80)
    logger.info(f"🚀 [프로세스 {args.process.upper()}] 시작")
    logger.info(f"  → 실행 스크립트: {script_to_run}")
    logger.info("="*80)

    # 5. 최종 명령어 생성
    base_cmd = [sys.executable, script_to_run]
    
    # 'p1and2'는 특별히 --data-type GK2A_PAST 인자를 강제로 추가
    if args.process == 'p1and2':
        cmd = base_cmd + ['--data-type', 'GK2A_PAST'] + unknown_args
    else:
        cmd = base_cmd + unknown_args

    # 6. 자식 스크립트 실행
    logger.info(f"실행 명령어: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, text=True, encoding='utf-8')
        
        logger.info("="*80)
        logger.info(f"🎉 [프로세스 {args.process.upper()}] 성공적으로 완료")
        logger.info("="*80)

    except subprocess.CalledProcessError as e:
        logger.error("="*80)
        logger.error(f"❌ [프로세스 {args.process.upper()}] 실행 중 오류 발생")
        logger.error(f"  → 반환 코드: {e.returncode}")
        logger.error("="*80)
        sys.exit(e.returncode)
    except FileNotFoundError:
        logger.error(f"❌ 오류: '{script_to_run}' 스크립트 파일을 찾을 수 없습니다.")
        logger.error("  → `scripts` 폴더에 모든 .py 파일이 있는지 확인하세요.")
        sys.exit(1)

if __name__ == "__main__":
    main()