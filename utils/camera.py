import cv2

def get_camera_resolution(cam_id=0):
    """카메라를 잠시 열어 실제 해상도(width, height)를 반환하는 함수"""
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return None, None
    
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
    else:
        # 읽기 실패 시 기본값이라도 가져오기 시도
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    cap.release()
    return w, h