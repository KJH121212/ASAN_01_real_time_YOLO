import yaml
from pathlib import Path

def load_exercise_configs(config_dir="configs/exercises"):
    """
    지정된 디렉토리 내의 모든 운동 설정(.yaml) 파일을 읽어
    하나의 통합된 파이썬 딕셔너리로 반환합니다.
    """
    exercise_config_dict = {}
    
    # Path(__file__).parent.parent 는 'utils' 폴더의 부모인 프로젝트 루트 폴더를 가리킵니다.
    base_path = Path(__file__).parent.parent / config_dir
    
    # 설정 폴더가 존재하는지 확인
    if not base_path.exists():
        print(f"[경고] 운동 설정 폴더를 찾을 수 없습니다: {base_path}")
        return exercise_config_dict
        
    # 폴더 내의 모든 .yaml 파일 순회
    for yaml_file in base_path.glob("*.yaml"):
        with open(yaml_file, 'r', encoding='utf-8') as f:
            try:
                # YAML 파일 읽기 (안전한 파싱)
                data = yaml.safe_load(f)
                
                # 파일 안에 'name' 키가 정상적으로 존재하는지 확인 (예: name: "biceps_curl")
                if data and 'name' in data:
                    ex_name = data['name']
                    # 해당 운동 이름(key)으로 뷰(views) 하위의 모든 설정값을 저장
                    exercise_config_dict[ex_name] = data.get('views', {})
                else:
                    print(f"[경고] {yaml_file.name} 파일에 'name' 속성이 없습니다. 무시됩니다.")
                    
            except yaml.YAMLError as e:
                print(f"[오류] {yaml_file.name} 파싱 중 에러 발생: {e}")
                
    return exercise_config_dict