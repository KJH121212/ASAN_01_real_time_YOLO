# ==============================================================================
# [파일 정보]
# 파일명: utils/normalization.py
# 작성자: 개발자 (Developer)
# 설명: 픽셀 좌표 기준 골반 원점 이동 및 체형 비율 100% 보존 등방성 정규화 유틸리티
# ==============================================================================

import numpy as np  # 유클리드 거리 스케일 및 다차원 배열 연산을 위한 모듈 로드


def normalize_pelvis_centered_pixel(keypoints: list, img_w: int, img_h: int) -> list:  # 픽셀 좌표를 등방성 상대 좌표로 변환하는 함수
    if not keypoints or len(keypoints) < 17:  # 유효하지 않은 입력 데이터 처리
        return keypoints  # 가공 없이 즉시 원본 반환

    kpt_map = {}  # 픽셀 복원 및 매핑용 딕셔너리 할당
    for kp in keypoints:  # 17개 관절 순회
        px = kp["x"] * img_w  # 해상도 비율 좌표를 다시 원본 픽셀 X 좌표(px)로 완전 복원
        py = kp["y"] * img_h  # 해상도 비율 좌표를 다시 원본 픽셀 Y 좌표(py)로 완전 복원
        kpt_map[kp["id"]] = (px, py, kp.get("score", 0.0))  # ID를 키로 하여 복원된 픽셀 튜플 저장

    CONF_THRES = 0.35  # 기준 관절 감지 유효성 최하 신뢰도 컷오프 세팅

    has_hips = (kpt_map.get(11, (0, 0, 0))[2] > CONF_THRES and kpt_map.get(12, (0, 0, 0))[2] > CONF_THRES)  # 좌/우 골반 픽셀 유효 감지 여부
    has_shoulders = (kpt_map.get(5, (0, 0, 0))[2] > CONF_THRES and kpt_map.get(6, (0, 0, 0))[2] > CONF_THRES)  # 좌/우 어깨 픽셀 유효 감지 여부

    origin_px, origin_py = 0.0, 0.0  # 원점으로 삼을 픽셀 기준점 초기화
    scale_px = 1.0  # 정규화를 위해 분모로 사용할 픽셀 단위 척도 초기화

    if has_hips and has_shoulders:  # 가장 이상적인 전신 감지 상황일 때
        hip_px = (kpt_map[11][0] + kpt_map[12][0]) / 2.0  # 좌우 골반 픽셀 X 중앙값 연산
        hip_py = (kpt_map[11][1] + kpt_map[12][1]) / 2.0  # 좌우 골반 픽셀 Y 중앙값 연산
        sh_px = (kpt_map[5][0] + kpt_map[6][0]) / 2.0  # 좌우 어깨 픽셀 X 중앙값 연산
        sh_py = (kpt_map[5][1] + kpt_map[6][1]) / 2.0  # 좌우 어깨 픽셀 Y 중앙값 연산
        origin_px, origin_py = hip_px, hip_py  # 골반 중앙 픽셀 위치를 영점(0,0)으로 확정
        scale_px = np.sqrt((sh_px - hip_px) ** 2 + (sh_py - hip_py) ** 2)  # 어깨 중심과 골반 중심의 픽셀 거리를 체형 척도로 산출

    elif has_shoulders:  # 하반신 가림으로 어깨만 감지된 차선책 상황일 때
        sh_px = (kpt_map[5][0] + kpt_map[6][0]) / 2.0  # 좌우 어깨 픽셀 X 중앙값 연산
        sh_py = (kpt_map[5][1] + kpt_map[6][1]) / 2.0  # 좌우 어깨 픽셀 Y 중앙값 연산
        origin_px, origin_py = sh_px, sh_py  # 어깨 중앙 픽셀 위치를 대체 영점(0,0)으로 확정
        scale_px = np.sqrt((kpt_map[5][0] - kpt_map[6][0]) ** 2 + (kpt_map[5][1] - kpt_map[6][1]) ** 2)  # 양측 어깨 너비를 픽셀 척도로 대체 산출

    else:  # 골반, 어깨 모두 가려진 치명적 결측 상황일 때
        return keypoints  # 정규화 불가 판정 후 원형 데이터 반환

    if scale_px < 1e-6:  # 척도가 수학적으로 너무 작아 분모 에러가 발생할 가능성이 있다면
        scale_px = 1.0  # 강제로 1.0 보정 적용

    normalized = []  # 결과물을 담을 빈 배열 생성
    for kp in keypoints:  # 다시 17개 관절 순회
        px = kp["x"] * img_w  # 픽셀 X 좌표 스케일 아웃
        py = kp["y"] * img_h  # 픽셀 Y 좌표 스케일 아웃
        norm_x = (px - origin_px) / scale_px  # X좌표에서 기준 원점을 차감하고 척도로 나누어 비율 왜곡(홀쭉해짐) 원천 차단
        norm_y = (py - origin_py) / scale_px  # Y좌표에서 기준 원점을 차감하고 척도로 나누어 체형 비율 유지 등방성 정규화 적용
        normalized.append({  # 딕셔너리 포장
            "id": kp["id"],  # 번호 속성 유지
            "x": round(float(norm_x), 4),  # 정규화 좌표 반올림 렌더링
            "y": round(float(norm_y), 4),  # 정규화 좌표 반올림 렌더링
            "score": kp.get("score", 0.0)  # 신뢰도 점수 유지
        })  # 완료된 객체를 리스트에 누적

    return normalized  # 비율이 완벽하게 보존된 정규화 키포인트 
