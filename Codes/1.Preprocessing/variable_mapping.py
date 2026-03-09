"""
천리안 GK2A 위성 데이터 처리를 위한 변수명 및 플래그 매핑 정보
- NC 파일 변수명 -> API 변수명
- 퀄리티 플래그 원본명 -> 퀄리티 플래그 API 변수명
- 퀄리티 플래그 처리 규칙 (클라이언트 요청: "기존" 로직 기준)
"""

# 1. NC 파일 변수 경로 -> API 변수명 매핑
# (제공해주신 'nc 파일 변수 경로' 테이블 기준)
VAR_MAP = {
    'CLD': {
        'CLD': 'cld'
    },
    'CTPS': {
        'CP': 'ctps_cp',
        'CTH': 'ctps_cth',
        'CTP': 'ctps_ctp',
        'CTT': 'ctps_ctt'
    },
    'CLA': {
        'CT': 'cla_type',
        'CF': 'cla_cloud_fraction'
    },
    'DCOEW': {
        'CER': 'dcoew_radius',
        'COT': 'dcoew_thickness', # 'dcoew_thickness' 오타 수정 (docew_thickness)
        'LWP': 'dcoew_liquid_path'
    },
    'NCOT': {
        'NCOT': 'ncot'
    },
    'CI': {
        'CI1': 'ci_ci1',
        'CI1_CCM': 'ci_ci1_ccm',
        'CI1_OBJ': 'ci_ci1_obj',
        'CI1_prob': 'ci_ci1_prob',
        'CI2': 'ci_ci2',
        'CI2_OBJ': 'ci_ci2_obj'
    },
    'FOG': {
        'FOG': 'fog'
        # 'fog_del_fta' 등은 원본 nc에 FOG/FOG 외 다른 변수가 있다면 추가
    },
    'RR': {
        'RR': 'rr'
    },
    'QPN': {
        'PAR': 'qpn_rate',
        'POR': 'qpn_probability'
    },
    'TQPROF': {
        'Q_profile': 'tqprof_q',
        'T_profile': 'tqprof_t'
        # pressure_levels 등은 API 변수명 테이블에 없으므로 제외
    },
    'TPW': {
        'LPW1': 'tpw_low',
        'LPW2': 'tpw_mid',
        'LPW3': 'tpw_high',
        'TPW': 'tpw'
    },
    'AII': {
        'CAPE': 'aii_cape',
        'KI': 'aii_ki',
        'LI': 'aii_li',
        'SI': 'aii_si',
        'TTI': 'aii_tti'
    },
    'APPS': {
        'AEP': 'apps_aep',
        'AOD': 'apps_aod',
        'DAOD055': 'apps_daod055',
        'DAOD11': 'apps_daod11'
    },
    'LST': {
        'LST': 'lst'
    },
    'SAL': {
        'BSA': 'sal_bsa',
        'BSA_B01': 'sal_bsa_b01',
        'BSA_B02': 'sal_bsa_b02',
        'BSA_B03': 'sal_bsa_b03',
        'BSA_B04': 'sal_bsa_b04',
        'BSA_B06': 'sal_bsa_b06',
        'WSA': 'sal_wsa',
        'WSA_B01': 'sal_wsa_b01',
        'WSA_B02': 'sal_wsa_b02',
        'WSA_B03': 'sal_wsa_b03',
        'WSA_B04': 'sal_wsa_b04',
        'WSA_B06': 'sal_wsa_b06'
    },
    'VGT': {
        'NDVI': 'vgt_ndvi',
        'EVI': 'vgt_evi'
        # 'FVC'는 API 변수명 테이블에 없으므로 제외
    },
    'SWRAD': {
        'DSR': 'swrad_downward',
        'ASR': 'swrad_absorbed'
    },
    'LWRAD': {
        'DLR': 'lwrad_downward',
        'ULR': 'lwrad_upward'
    }
}


# 2. 퀄리티 플래그 원본명 -> API 변수명 매핑
# (제공해주신 '이후 변수 퀄리티 플래그 처리 방법' 테이블 기준)
# 키: 폴더명(소문자), 값: { 원본 DQF명: 최종 API DQF명 }
FLAG_MAP = {
    'aii': {
        'quality_flag1': 'aii_dqf1',
        'quality_flag2': 'aii_dqf2',
        'quality_flag3': 'aii_dqf3'
    },
    'apps': {
        'AEP_DQF': 'apps_aep_dqf',
        'AOD_DQF': 'apps_aod_dqf',
        'DAOD055_DQF': 'apps_daod055_dqf',
        'DAOD11_DQF': 'apps_daod11_dqf'
    },
    'ci': {
        'DQF_CI1': 'ci_ci1_dqf'
    },
    'cla': {
        'CA_DQF': 'cla_dqf1',
        'CF_DQF': 'cla_cloud_fraction_dqf',
        'CT_DQF': 'cla_type_dqf'
    },
    'cld': {
        # 'cld'는 플래그 없음
    },
    'ctps': {
        'CLD_EMIS_11_flag': 'ctps_dqf2',
        'CP_flag': 'ctps_cp_dqf',
        'CTH_flag': 'ctps_cth_dqf',
        'CTPS_flag': 'ctps_dqf1', # 핵심 플래그
        'CTP_flag': 'ctps_dqf5',
        'CTT_flag': 'ctps_ctt_dqf'
    },
    'dcoew': {
        'DCOEW_DQF': 'dcoew_dqf1' # 핵심 플래그
    },
    'lst': {
        'DQF_LST': 'lst_dqf' # 핵심 플래그
    },
    'lwrad': {
        'BEMIS_DQF': 'lwrad_dqf2',
        'DLR_DQF': 'lwrad_downward_dqf', # DLR_DQF1이 아님 (표 기준)
        'LWRAD_DQF': 'lwrad_dqf1',
        'OLR_DQF': 'lwrad_dqf3',
        'ULR_DQF': 'lwrad_upward_dqf'
    },
    'ncot': {
        'NCOT_DQF': 'ncot_dqf' # 핵심 플래그
    },
    'qpn': {
        'QPN_DQF': 'qpn_dqf1'
    },
    'rr': {
        'Raining_CT_flag': 'rr_raining_ct_flag'
    },
    'sal': {
        'DQF_BSA': 'sal_dqf1', # 핵심 플래그
        'DQF_WSA': 'sal_dqf2'  # 핵심 플래그
    },
    'swrad': {
        'ASR_DQF1': 'swrad_absorbed_dqf',
        'DSR_DQF1': 'swrad_downward_dqf',
        'RSR_DQF1': 'swrad_upward_dqf',
        'SW_DQF': 'swrad_dqf1'
    },
    'tpw': {
        'quality_flag1': 'tpw_dqf1',
        'quality_flag2': 'tpw_dqf2',
        'quality_flag3': 'tpw_dqf3'
    },
    'tqprof': {
        'quality_flag1': 'tqprof_dqf1',
        'quality_flag2': 'tqprof_dqf2',
        'quality_flag3': 'tqprof_dqf3'
    },
    'vgt': {
        'DQF': 'vgt_dqf1' # 핵심 플래그
    },
    'fog': {
        'DQF_FOG': 'fog_dqf' # 핵심 플래그
    }
}


# 3. 퀄리티 플래그 처리 규칙 (클라이언트 요청: "기존" 로직 기준)
# (제공해주신 '이전 변수 퀄리티 플래그 처리 방법' 테이블 기준)
#
# 참고: 이 규칙을 사용하려면 process_all_sample.py의
# `clean_outliers` 함수를 이 DQF_RULES를 사용하도록 수정해야 합니다.
# 현재 `clean_outliers` 함수는 이 로직을 하드코딩하고 있습니다.

DQF_RULES = {
    # 키: 폴더명(대문자)
    'VGT': {
        'flag_name': 'DQF', # 원본 플래그명 (FLAG_MAP을 통해 'vgt_dqf1'로 매핑됨)
        'rule_type': 'BITMASK',
        'rules': {
            # '유효' 조건 (0이어야 함)
            'AHI_disk': {'bit': 7, 'value': 0},
            'FVC_quality': {'bit': 5, 'value': 0, 'target_vars': ['vgt_fvc']},
            'EVI_quality': {'bit': 4, 'value': 0, 'target_vars': ['vgt_evi']},
            'NDVI_quality': {'bit': 3, 'value': 0, 'target_vars': ['vgt_ndvi']},
            'Land_Water': {'bit': 2, 'value': 0} # 0=Land
        }
    },
    'TQPROF': {
        'rule_type': 'AND_EQUALS',
        'flags': ['quality_flag1', 'quality_flag2'],
        'values': [0, 1]
    },
    'TPW': {
        'rule_type': 'AND_EQUALS',
        'flags': ['quality_flag1', 'quality_flag2'],
        'values': [0, 1]
    },
    'SWRAD': {
        'rule_type': 'EQUALS',
        'flag_prefix': '_DQF', # SWRAD/DSR -> DSR_DQF, SWRAD/ASR -> ASR_DQF
        'value': 1
    },
    'SAL': {
        'rule_type': 'EQUALS',
        'flag_prefix': 'DQF_', # SAL/BSA -> DQF_BSA, SAL/WSA -> DQF_WSA
        'value': 1
    },
    'NCOT': {
        'rule_type': 'EQUALS',
        'flag_name': 'NCOT_DQF',
        'value': 0
    },
    'LWRAD': {
        'rule_type': 'EQUALS',
        'flag_prefix': '_DQF', # LWRAD/DLR -> DLR_DQF, LWRAD/ULR -> ULR_DQF
        'value': 1
    },
    'LST': {
        'rule_type': 'EQUALS',
        'flag_name': 'DQF_LST',
        'value': 0
    },
    'FOG': {
        'rule_type': 'EQUALS',
        'flag_name': 'DQF_FOG',
        'value': 0
    },
    'DCOEW': {
        'rule_type': 'EQUALS',
        'flag_name': 'DCOEW_DQF',
        'value': 0
    },
    'CTPS': {
        'rule_type': 'EQUALS',
        'flag_name': 'CTPS_flag',
        'value': 0
    },
    'CLA': { # CLD--CLA
        'rule_type': 'EQUALS',
        'flag_prefix': '_DQF', # CLA/CT -> CT_DQF, CLA/CF -> CF_DQF
        'value': 0
    },
    'APPS': {
        'rule_type': 'EQUALS',
        'flag_prefix': '_DQF', # APPS/AEP -> AEP_DQF, APPS/AOD -> AOD_DQF
        'value': 2
    },
    'AII': {
        'rule_type': 'AND_EQUALS',
        'flags': ['quality_flag1', 'quality_flag2'],
        'values': [0, 1]
    },
    
    # '기존' 규칙에 없던 항목 (플래그 없음 또는 기본값)
    'RR': { 'rule_type': 'NONE' },
    'QPN': { 'rule_type': 'NONE' },
    'CI': { 'rule_type': 'NONE' },
    'CLD': { 'rule_type': 'NONE' }
}


# ========== 스크립트에서 호출할 함수 ==========

def get_variable_mapping(folder_name: str) -> dict:
    """
    폴더명(대문자)에 해당하는 [NC 변수명 -> API 변수명] 매핑 반환
    예: get_variable_mapping('CTPS') -> {'CP': 'ctps_cp', 'CTH': 'ctps_cth', ...}
    """
    return VAR_MAP.get(folder_name.upper(), {})

def get_flag_mapping(folder_name: str) -> dict:
    """
    폴더명(소문자)에 해당하는 [원본 플래그명 -> API 플래그명] 매핑 반환
    예: get_flag_mapping('aii') -> {'quality_flag1': 'aii_dqf1', ...}
    """
    return FLAG_MAP.get(folder_name.lower(), {})

def get_dqf_rule(folder_name: str) -> dict:
    """
    폴더명(대문자)에 해당하는 "기존" DQF 처리 규칙 반환
    예: get_dqf_rule('LST') -> {'rule_type': 'EQUALS', 'flag_name': 'DQF_LST', 'value': 0}
    """
    return DQF_RULES.get(folder_name.upper(), {'rule_type': 'NONE'})

# ========== (참고) 매핑 적용을 위한 로직 (예시) ==========
# 
# 현재 process_all_sample.py는 이 로직을 사용하지 않고,
# clean_outliers 함수에 하드코딩되어 있습니다.
# 만약 매핑 파일을 사용하도록 스크립트를 리팩토링한다면
# 아래와 같은 함수들이 필요합니다.
# 
def apply_variable_mapping(df, folder_name):
    """
    DataFrame의 컬럼명을 NC 기준에서 API 기준으로 변경
    (process_all_sample.py의 804라인에서 호출 시도하는 함수)
    """
    var_map = get_variable_mapping(folder_name)
    flag_map = get_flag_mapping(folder_name)
    
    # 변수명과 플래그명 모두 매핑
    full_map = {**var_map, **flag_map}
    
    # 매핑할 컬럼만 필터링
    rename_map = {old: new for old, new in full_map.items() if old in df.columns}
    
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        
    return df
    
def get_final_columns():
    """
    최종 산출물 CSV (gk2A최종 결과물형식 - 시트1.csv) 기준
    컬럼명과 순서를 반환합니다. (총 88개)
    """
    return [
        'geoId', 'geo_lon', 'geo_lat', 'dateTime', 'cld', 'ctps_cp', 'ctps_cth',
        'ctps_ctp', 'ctps_ctt', 'cla_type', 'cla_cloud_fraction',
        'dcoew_thickness', 'dcoew_radius', 'dcoew_liquid_path', 'ncot', 'ci_ci1',
        'ci_ci1_ccm', 'ci_ci1_obj', 'ci_ci1_prob', 'ci_ci2', 'ci_ci2_obj',
        'fog', 'rr', 'qpn_rate', 'qpn_probability', 'tqprof_t', 'tqprof_q',
        'tpw', 'tpw_low', 'tpw_mid', 'tpw_high', 'aii_cape', 'aii_ki', 'aii_li',
        'aii_si', 'aii_tti', 'apps_aod', 'apps_daod11', 'apps_daod055',
        'apps_aep', 'lst', 'sal_bsa', 'sal_bsa_b01', 'sal_bsa_b02',
        'sal_bsa_b03', 'sal_bsa_b04', 'sal_bsa_b06', 'sal_wsa', 'sal_wsa_b01',
        'sal_wsa_b02', 'sal_wsa_b03', 'sal_wsa_b04', 'sal_wsa_b06',
        'vgt_ndvi', 'vgt_evi', 'swrad_absorbed', 'swrad_downward',
        'lwrad_downward', 'lwrad_upward', 'aii_dqf1', 'aii_dqf2',
        'apps_aep_dqf', 'apps_aod_dqf', 'apps_daod055_dqf', 'apps_daod11_dqf',
        'cla_cloud_fraction_dqf', 'cla_type_dqf', 'ctps_dqf1', 'dcoew_dqf1',
        'lst_dqf', 'lwrad_downward_dqf', 'lwrad_dqf1', 'lwrad_upward_dqf',
        'ncot_dqf', 'qpn_dqf1', 'rr_raining_ct_flag', 'sal_dqf1', 'sal_dqf2',
        'swrad_absorbed_dqf', 'swrad_downward_dqf', 'swrad_dqf1', 'tpw_dqf1',
        'tpw_dqf2', 'tqprof_dqf1', 'tqprof_dqf2', 'vgt_dqf1', 'fog_dqf'
    ]

def get_daily_fill_columns():
    """
    하루 1회(00시) 수집되어
    모든 시간에 채워넣어야 하는 변수 목록
    (SAL, VGT 변수들)
    """
    return [
        'sal_bsa', 'sal_bsa_b01', 'sal_bsa_b02', 'sal_bsa_b03', 'sal_bsa_b04', 'sal_bsa_b06',
        'sal_wsa', 'sal_wsa_b01', 'sal_wsa_b02', 'sal_wsa_b03', 'sal_wsa_b04', 'sal_wsa_b06',
        'vgt_ndvi', 'vgt_evi',
        # 해당 DQF 플래그도 값과 함께 채워넣습니다.
        'sal_dqf1', 'sal_dqf2', 'vgt_dqf1'
    ]