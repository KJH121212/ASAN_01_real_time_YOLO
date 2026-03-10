import numpy as np

def normalize_realtime_12kpts(kpts_12):
    """
   의 normalize_skeleton_array 로직을 실시간 프레임용으로 이식
    - kpts_12: 얼굴이 제거된 (12, 3) 형태의 넘파이 배열
    """
    norm_kpts = kpts_12.copy().astype(float)
    
    # 모든 값이 0이면 (인식 실패) 그대로 반환
    if np.all(norm_kpts == 0):
        return norm_kpts
    
    # 1. 중앙점 계산: 골반 중심 (6번, 7번 중점)
    # cropped_kps 기준: 6(왼쪽 골반), 7(오른쪽 골반)
    hip_center = (norm_kpts[6, :2] + norm_kpts[7, :2]) / 2.0
    
    # 2. 기준 거리 계산: 몸통 길이 (어깨 중점과 골반 중점 사이 거리)
    # cropped_kps 기준: 0(왼쪽 어깨), 1(오른쪽 어깨)
    shoulder_center = (norm_kpts[0, :2] + norm_kpts[1, :2]) / 2.0
    torso_length = np.linalg.norm(shoulder_center - hip_center)
    
    # 3. 정규화 실행: 모든 좌표에서 hip_center를 빼고 torso_length로 나눔
    if torso_length > 1e-6:
        norm_kpts[:, :2] = (norm_kpts[:, :2] - hip_center) / torso_length
        
    return norm_kpts