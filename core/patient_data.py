"""PatientData 공통 식별자 규칙입니다. Unity의 등록 예약 파일은 수정하지 않습니다."""

# 버전과 공통 ID에 경로 구분자가 들어오지 않도록 검사합니다.
import re
# Unity 연결 없이 테스트할 때도 충돌하지 않는 ID를 만듭니다.
import uuid

# Unity BuildInfo의 최종 버전 폴더명입니다. 연동 시 Unity가 실제 값을 전달해야 합니다.
DEFAULT_DATA_VERSION = 'v1.0.0(0)'


# 일반 파일 ID와 달리 버전의 점·괄호는 그대로 유지합니다.
def version_folder(value):
    # 환경명이나 경로가 아닌 최종 버전 문자열만 허용합니다.
    if not isinstance(value, str) or not re.fullmatch(r'v\d+\.\d+\.\d+\(\d+\)', value):
        raise ValueError('data_version must look like v1.0.0(0)')
    # v2.3.4(27)을 v2.3.4_27_로 바꾸지 않습니다.
    return value


# 등록 차수는 Unity가 발급한 값을 사용하며 서버가 자동 증가시키지 않습니다.
def registration_number(value):
    # 정수 1과 문자열 01을 같은 등록 차수로 처리합니다.
    if isinstance(value, str) and re.fullmatch(r'\d{1,2}', value):
        value = int(value)
    # bool·0·음수·소수는 등록 번호가 아닙니다.
    if type(value) is not int or not 1 <= value <= 99:
        raise ValueError('registration must be an integer from 1 to 99')
    # 디렉터리 이름에 두 자리를 유지합니다.
    return f'{value:02d}'


# 플레이·운동 ID를 파일 경로의 안전한 단일 구성 요소로 검증합니다.
def shared_identifier(value, name):
    # 단독 테스트에서 생략한 ID는 새로 생성합니다.
    if value is None:
        return uuid.uuid4().hex
    # 실제 ID와 설명용 짧은 ID를 허용하되 경로 구분자와 빈 문자열은 거부합니다.
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value):
        raise ValueError(f'{name} must contain only letters, numbers, underscores or hyphens')
    # Windows에서 파일명으로 사용할 수 없는 장치 이름을 차단합니다.
    if value.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        raise ValueError(f'{name} cannot be a reserved device name')
    # 원래 ID를 변환하지 않아 Unity 기록과 정확히 연결할 수 있습니다.
    return value
