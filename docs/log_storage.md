# 로그 저장 위치

기준은 `PatientData_사용안내.pdf`이며, Unity와 공통 폴더·ID를 사용하고 카메라 전용 로그를 저장합니다. 전체 구조·입력 JSON·항목 설명은 [PatientData 카메라 로그 규격](patient_data_camera.md)을 참고하세요.

기본 폴더는 Python 실행 시 프로젝트의 `PatientData`, exe 실행 시 exe 옆 `PatientData`입니다. 우선순위는 시작 JSON의 `log_root` → 서버 환경 변수 `ASAN_LOG_ROOT` → 기본 폴더입니다. `log_root`에는 버전·환자 폴더가 아닌 PatientData 최상위 경로를 지정합니다.

```text
PatientData/버전/환자번호/등록차수/
├─ Sensors/수집시작일/플레이ID/Camera/
│  ├─ R01_S02_Set01_Play_Camera_파일ID.csv
│  ├─ R01_S02_Set01_Play_Camera_파일ID.npz
│  ├─ R01_S02_Set01_Play_Camera_파일ID.mp4
│  ├─ R01_S02_Set01_Play_Camera_파일ID_reps.csv
│  └─ R01_S02_Set01_Play_Camera_파일ID.json
└─ Management/Camera/
   ├─ profile.json
   ├─ patient_metadata.csv
   ├─ Records/파일ID.json
   └─ Daily/측정종료일.csv
```

Unity에서 `data_version`, `registration`, `play_session_id`, `exercise_record_id`, `slot`을 추가로 전달합니다. 같은 플레이의 ID와 같은 운동 기록의 ID를 여러 세트에서 유지하세요. 응답의 기존 `session_id`는 카메라 측정 구간 ID로 유지합니다.

원시 파일 날짜는 시작일 KST, 카메라 요약 날짜는 시작 시각에 측정 경과 시간을 더한 종료일 KST입니다. 좌표·길이는 몸통 길이 정규화 단위, 각도는 도 단위로 기록합니다. IMU 가속도·자이로 값은 생성하지 않습니다.

유효한 입력 영상은 항상 MP4로 기록하며 준비 시간에는 추론·녹화하지 않습니다. 프레임 CSV는 매 분석 프레임에 저장하고, NPZ·반복 CSV·요약은 종료 시 확정합니다. 강제 종료 시 미완성 영상이나 저장 전 NPZ의 복구를 보장하지 않습니다. PDF의 Unity 복구 큐·엑셀 생성은 Unity가 관리합니다.

기존 `data`와 구형 버전 폴더는 자동 이동·병합·삭제하지 않습니다. Unity의 PlayData·Records·등록 예약·IMU 복구 자료는 Python 서버가 변경하지 않습니다. 서버를 재시작해야 새 경로가 적용됩니다.
