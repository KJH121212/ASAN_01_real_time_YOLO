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


def normalize_pelvis_centered(keypoints: list) -> list:
  """관절 감지 상태에 따라 원점과 스케일을 가변 변환하는 계층적 정규화 함수"""
  if not keypoints or len(keypoints) < 17:
    return keypoints

  kpt_map = {
      kp["id"]: (kp["x"], kp["y"], kp.get("score", 0.0)) for kp in keypoints
  }

  CONF_THRES = 0.35  # 관절 인식 신뢰도 임계값

  # 필수 관절 인식 여부 검사
  has_hips = (
      kpt_map.get(11, (0, 0, 0))[2] > CONF_THRES
      and kpt_map.get(12, (0, 0, 0))[2] > CONF_THRES
  )
  has_shoulders = (
      kpt_map.get(5, (0, 0, 0))[2] > CONF_THRES
      and kpt_map.get(6, (0, 0, 0))[2] > CONF_THRES
  )

  origin_x, origin_y = 0.0, 0.0
  scale = 1.0

  # [1순위] 골반과 어깨가 모두 잘 보일 때 (골반 중심 + 상체 길이 스케일)
  if has_hips and has_shoulders:
    hip_x = (kpt_map[11][0] + kpt_map[12][0]) / 2.0
    hip_y = (kpt_map[11][1] + kpt_map[12][1]) / 2.0
    sh_x = (kpt_map[5][0] + kpt_map[6][0]) / 2.0
    sh_y = (kpt_map[5][1] + kpt_map[6][1]) / 2.0

    origin_x, origin_y = hip_x, hip_y
    scale = np.sqrt((sh_x - hip_x) ** 2 + (sh_y - hip_y) ** 2)

  # [2순위] 하체가 가려지고 상체만 보일 때 (어깨 중심 + 어깨 폭 스케일)
  elif has_shoulders:
    sh_x = (kpt_map[5][0] + kpt_map[6][0]) / 2.0
    sh_y = (kpt_map[5][1] + kpt_map[6][1]) / 2.0

    origin_x, origin_y = sh_x, sh_y
    scale = np.sqrt(
        (kpt_map[5][0] - kpt_map[6][0]) ** 2
        + (kpt_map[5][1] - kpt_map[6][1]) ** 2
    )

  # [3순위] 주요 관절 모두 미감지 시 원본 유지
  else:
    return keypoints

  if scale < 1e-6:
    scale = 1.0

  # 좌표 변환 진행
  normalized = []
  for kp in keypoints:
    norm_x = (kp["x"] - origin_x) / scale
    norm_y = (kp["y"] - origin_y) / scale
    normalized.append({
        "id": kp["id"],
        "x": round(float(norm_x), 4),
        "y": round(float(norm_y), 4),
        "score": kp.get("score", 0.0),
    })

  return normalized