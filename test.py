import pandas as pd
import os

# 1. 파일 경로 설정 및 CSV 로드
file_path = './data/metadata_v3.0.csv' if os.path.exists('./data/metadata_v3.0.csv') else 'metadata_v3.0.csv'

df = pd.read_csv(file_path)

# 2. common_path 컬럼에서 'game' 또는 'gametest' 키워드가 들어간 행 필터링
game_df = df[df['common_path'].astype(str).str.contains('game', case=False, na=False)].copy()

# 3. common_path 파일명 파싱 함수 정의
#    형식 예시: game_test/norm_angle_cam/frontal__alternating_biceps_curl__incline1
def parse_game_path(path):
    parts = path.split('/')
    subfolder = parts[1] if len(parts) > 2 else ''  # norm_angle_cam, wide_angle_cam 등
    filename = parts[-1]
    
    # 파일명 구조 split ('__' 기준): [시점, 동작명, 세부옵션]
    fn_parts = filename.split('__')
    view = fn_parts[0] if len(fn_parts) > 0 else ''
    exercise = fn_parts[1] if len(fn_parts) > 1 else ''
    detail = fn_parts[2] if len(fn_parts) > 2 else ''
    
    return pd.Series({
        'subfolder': subfolder,
        'view': view,
        'exercise': exercise,
        'detail': detail,
        'filename': filename
    })

# 4. 파싱 데이터 적용 및 데이터프레임 병합
parsed_df = game_df['common_path'].apply(parse_game_path)
eda_result = pd.concat([game_df[['common_path']], parsed_df], axis=1)

# 5. EDA 결과 출력
print(f"==================================================")
print(f"📊 [game_test] 총 데이터 수: {len(eda_result)}개")
print(f"==================================================\n")

print("📌 1. 전체 포함된 동작(Exercise) 종류 및 개수:")
exercise_counts = eda_result['exercise'].value_counts()
print(exercise_counts)

print("\n--------------------------------------------------")
print("📌 2. 촬영 환경(Subfolder)별 동작 분포:")
print(pd.crosstab(eda_result['exercise'], eda_result['subfolder']))

print("\n--------------------------------------------------")
print("📌 3. 파싱 샘플 데이터 (상위 5개):")
print(eda_result[['subfolder', 'view', 'exercise', 'detail']].head())