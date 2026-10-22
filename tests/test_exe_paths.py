"""실제 exe 빌드 없이 배포 환경의 기본 로그 경로를 검증합니다."""

# 경로와 환경 변수만 바꿔 모델·카메라 없이 검증합니다.
from pathlib import Path
# 표준 테스트 실행기를 사용합니다.
import unittest
# 실행 파일 환경을 일시적으로 흉내 내기 위한 패치 도구입니다.
from unittest.mock import patch
# 실제 로그 기본 경로 계산 함수를 검증합니다.
from core.session_log import default_log_root


# exe와 Python 실행에서 경로 우선순위를 확인합니다.
class ExePathTests(unittest.TestCase):
    # 배포본은 _internal이나 임시 폴더가 아닌 exe 옆에 데이터를 기록해야 합니다.
    def test_frozen_logs_are_next_to_exe(self):
        # 테스트 경로 문자열만 사용하며 실제 폴더를 만들지 않습니다.
        executable = str(Path('dist/ASANSocketServer/ASANSocketServer.exe').resolve())
        # PyInstaller의 실행 플래그와 실행 경로를 모의합니다.
        with patch('core.session_log.sys.frozen', True, create=True), patch('core.session_log.sys.executable', executable), patch.dict('os.environ', {}, clear=True):
            self.assertEqual(default_log_root(), Path(executable).parent / 'data')

    # 사용자가 지정한 저장 경로는 exe의 기본 경로보다 우선합니다.
    def test_environment_path_overrides_frozen_default(self):
        # 기존 환경을 수정하지 않고 해당 호출 안에서만 변수를 바꿉니다.
        with patch('core.session_log.sys.frozen', True, create=True), patch.dict('os.environ', {'ASAN_LOG_ROOT': 'custom_logs'}):
            self.assertEqual(default_log_root(), Path('custom_logs').resolve())
