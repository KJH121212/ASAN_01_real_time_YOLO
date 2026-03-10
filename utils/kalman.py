# utils/kalman.py
import cv2
import numpy as np

class JointKalmanTracker:
    def __init__(self):
        # 4개 상태(x, y, dx, dy), 2개 측정값(x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # 측정 행렬: 상태에서 x, y만 추출
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                              [0, 1, 0, 0]], np.float32)
        
        # 전이 행렬: 이전 상태를 기반으로 다음 위치 예측 (등속 모델)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                             [0, 1, 0, 1], 
                                             [0, 0, 1, 0], 
                                             [0, 0, 0, 1]], np.float32)
        
        # 프로세스 노이즈 (클수록 예측보다 측정을 더 믿음)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False

    def update(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        
        if not self.initialized:
            # 초기 위치 설정
            self.kf.statePost = np.array([[meas[0,0]], [meas[1,0]], [0], [0]], np.float32)
            self.initialized = True
            return x, y
        
        # 1. 예측 (Predict)
        self.kf.predict()
        
        # 2. 보정 (Correct)
        est = self.kf.correct(meas)
        
        return est[0,0], est[1,0]