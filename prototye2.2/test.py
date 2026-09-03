import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_socket_server_pipeline(output_filename="socket_server_pipeline.png"):
    # 16:9 Ultra HD 해상도 캔버스 설정 (3200 x 1800 at 160 dpi)
    fig, ax = plt.subplots(figsize=(20, 11.25), facecolor="#0B0F19")
    ax.set_facecolor("#0B0F19")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    font_main = "DejaVu Sans"
    color_bg_card = "#131C2E"
    color_cyan = "#38BDF8"    # 입력 & 카메라
    color_indigo = "#818CF8"  # AI 추론 & 신호 필터링
    color_amber = "#F59E0B"   # 캘리브레이션 & 상태 분기
    color_emerald = "#10B981" # 본 운동 FSM & 패킷 송출
    color_rose = "#F43F5E"    # 가림 감지(Occlusion)
    color_slate = "#94A3B8"   # 보조 설명 텍스트

    # --- 상단 타이틀 ---
    ax.text(50, 97.0, "ExerciseSocketServer (socket_server.py) - Internal Execution Pipeline",
            fontsize=18, fontweight="bold", color="#FFFFFF", ha="center", fontname=font_main)
    ax.text(50, 94.6, "Async Client Connection ➔ Multi-threaded Ingestion ➔ Dual AI Stream ➔ State Machine Routing ➔ Network & Disk Dispatch",
            fontsize=10.2, color=color_slate, ha="center", fontname=font_main)

    # 모듈 카드 그리기 헬퍼
    def draw_box(x, y, w, h, title, subtitle=None, details=[], border_color="#38BDF8", fill_color=color_bg_card, title_color="#FFFFFF"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                     linewidth=1.3, edgecolor=border_color, facecolor=fill_color, zorder=2)
        ax.add_patch(rect)
        
        header_rect = patches.FancyBboxPatch((x, y + h - 3.8), w, 3.8, boxstyle="round,pad=0.08,rounding_size=0.8",
                                            linewidth=0, facecolor=border_color, alpha=0.18, zorder=3)
        ax.add_patch(header_rect)
        
        ax.text(x + w/2, y + h - 2.3, title, fontsize=10.5, fontweight="bold", color=title_color, ha="center", va="center", zorder=4, fontname=font_main)
        
        cur_y = y + h - 5.2
        if subtitle:
            ax.text(x + w/2, cur_y, subtitle, fontsize=8.2, color=border_color, fontweight="bold", ha="center", va="center", zorder=4, fontname=font_main)
            cur_y -= 2.6
            
        for d in details:
            ax.text(x + 1.2, cur_y, f"•  {d}", fontsize=7.8, color="#E2E8F0", ha="left", va="center", zorder=4, fontname=font_main)
            cur_y -= 2.2

    # --- 4개 단계별 배경 패널 ---
    stages = [
        (2.5, 5, 20.5, 85, "1. INGESTION & THREADING", color_cyan),
        (25.5, 5, 22.5, 85, "2. THROTTLING & POSE AI", color_indigo),
        (50.5, 5, 24.5, 85, "3. OCCLUSION & DUAL ROUTING", color_amber),
        (77.5, 5, 20.0, 85, "4. DISPATCH & STORAGE", color_emerald)
    ]

    for sx, sy, sw, sh, stitle, scolor in stages:
        bg = patches.FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    linewidth=1.0, edgecolor="#1E293B", facecolor="#0F172A", alpha=0.6, zorder=1)
        ax.add_patch(bg)
        ax.text(sx + sw/2, 87.2, stitle, fontsize=9.2, fontweight="bold", color=scolor, ha="center", zorder=3, fontname=font_main)

    # [1단계: 클라이언트 연결 및 스레드 카메라 프레임 수집]
    draw_box(4.2, 66, 17.0, 18, "handle_client()", "Async WebSocket Entry",
             ["Client connects via ws://...:8080", "Wait for JSON CMD_SET_SESSION", "init_session() parameter mapping", "Build Controller & Engines"],
             border_color=color_cyan)

    draw_box(4.2, 43, 17.0, 19, "ThreadedCamera", "Background Thread Loop",
             ["cv2.VideoCapture(CAP_DSHOW)", "self.cap.set(BUFFERSIZE, 1)", "threading.Lock() sync lock", "Continuous loop: self.cap.read()"],
             border_color=color_cyan)

    draw_box(4.2, 18, 17.0, 21, "camera.read()", "Non-blocking Safe Fetch",
             ["with self.read_lock:", "  frame = self.frame.copy()", "Calculate real-time loop FPS", "Discard frame lag / queue delay"],
             border_color=color_cyan)

    # [2단계: 영상 압축 및 병렬 AI 관절 추출]
    draw_box(27.0, 66, 19.5, 18, "JPEG Compression", "Network Throttling",
             ["if frame_count % 2 == 0:", "cv2.resize (400 x 300)", "cv2.imencode(.jpg, quality=30)", "Base64 string encoding (b64)", "asyncio.to_thread() async offload"],
             border_color=color_indigo)

    draw_box(27.0, 40, 19.5, 22, "SkeletonEngine", "Deep Learning Inference",
             ["asyncio.to_thread() offload", "YOLOv8 person detector (classes=[0])", "Interval=5 center BBox caching", "RTMPose top-down inference", "17 Keypoints (COCO norm [0..1])"],
             border_color=color_indigo)

    draw_box(27.0, 16, 19.5, 20, "RealtimeEMAFilter", "Signal Cleansing & Jitter Clamp",
             ["filter_engine.update(raw_kpts)", "Norm jump > 0.15: Vector clamp", "alpha=0.6 Exponential Moving Avg", "Prev frame cache fallback", "Output: smoothed 17 keypoints"],
             border_color=color_indigo)

    # [3단계: 전신 가림 판별 및 모드별 듀얼 라우팅]
    draw_box(52.0, 71, 21.5, 14, "check_occlusion()", "12 Essential Keypoints",
             ["Check 12 joints confidence >= 0.35", "Shoulders, elbows, hips, knees, ankles", "is_occluded flag calculation"],
             border_color=color_rose)

    draw_box(52.0, 43, 21.5, 24, "CALIBRATION Mode", "controller.process_calib()",
             ["If is_occluded: Rollback to FULL_BODY", "FULL_BODY_CHECK (Waiting body)", "COUNTDOWN (3.0s timer)", "COLLECTING: append values", "FINISHED: compute 85% ROM", "Send CALIBRATION_FINISHED"],
             border_color=color_amber)

    draw_box(52.0, 14, 21.5, 25, "MAIN (Test) Mode", "controller.process_main()",
             ["If is_occluded: Freeze last rep cache", "MotionEngine: Angle / Relative-Y", "SideFSM (READY->PUSH->WAIT)", "Quality: PERFECT / GOOD / BAD", "Target reps check -> is_finished", "Send SESSION_FINISHED"],
             border_color=color_emerald)

    # [4단계: 클라이언트 패킷 송출 및 데이터 영구 저장]
    draw_box(79.0, 52, 17.0, 33, "POSE_UPDATE", "Real-Time Packet Dispatch",
             ["Assemble JSON payload:", "  type: 'POSE_UPDATE'", "  mode, left, right status", "  keypoints (smoothed)", "  fps, is_occluded", "  frame_b64 (compressed)", "  calib_step, remaining_sec", "websocket.send(payload)"],
             border_color=color_emerald, title_color=color_emerald)

    draw_box(79.0, 12, 17.0, 36, "DataManager", "Session Data Persistence",
             ["_save_session_files() calls:", "1. {player_id}.json", "   - User custom ROM thresholds", "   - Calibration history", "2. rep_details.csv", "   - Rep#, Duration, ROM, Quality", "3. {session}_skeleton.npz", "   - 17kpts compressed time-series"],
             border_color=color_cyan)

    # 데이터 플로우 화살표 연결 헬퍼
    def draw_arrow(x1, y1, x2, y2, color="#38BDF8", label="", rad=0.0, label_pos=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45",
                                    color=color, lw=1.6, connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3))
        if label:
            if label_pos:
                lx, ly = label_pos
            else:
                lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 + 1.2
            ax.text(lx, ly, label, fontsize=7.2, color=color, fontweight="bold", ha="center", fontname=font_main,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#0B0F19", ec=color, lw=0.6, alpha=0.95), zorder=5)

    # 1. 수집 파이프라인
    draw_arrow(12.7, 66, 12.7, 62, color=color_cyan, label="start_camera()")
    draw_arrow(12.7, 43, 12.7, 39, color=color_cyan, label="cap.read()")

    # 2. 카메라 프레임 -> JPEG 분기와 AI 분기
    draw_arrow(21.2, 33, 27.0, 75, color=color_indigo, label="Frame", rad=0.18)
    draw_arrow(21.2, 27, 27.0, 51, color=color_indigo, label="Frame", rad=0.0)

    # 3. AI 추론 -> EMA 필터
    draw_arrow(36.75, 40, 36.75, 36, color=color_indigo, label="Raw Kpts")

    # 4. 필터 완료 관절 -> 가림 판별 및 분기
    draw_arrow(46.5, 26, 52.0, 78, color=color_rose, label="Kpts", rad=0.18)
    draw_arrow(62.75, 71, 62.75, 67, color=color_amber, label="if CALIBRATION")
    draw_arrow(52.0, 73, 52.0, 39, color=color_emerald, label="if MAIN", rad=-0.28, label_pos=(49.5, 55))

    # 5. JPEG Base64 캐시 -> 상단 우회 경로로 POSE_UPDATE 패킷 전달
    draw_arrow(46.5, 78, 79.0, 78, color=color_indigo, label="last_frame_b64 (cached)", rad=-0.32, label_pos=(62.75, 90.8))

    # 6. 모드별 연산 결과 -> 클라이언트 전송
    draw_arrow(73.5, 57, 79.0, 65, color=color_amber, label="Calib Step", rad=-0.05)
    draw_arrow(73.5, 29, 79.0, 58, color=color_emerald, label="Rep Stats", rad=0.06)

    # 7. 세션 종료 시 스토리지 저장
    draw_arrow(73.5, 49, 79.0, 36, color=color_amber, label="Thresholds", rad=0.08)
    draw_arrow(73.5, 21, 79.0, 24, color=color_emerald, label="Buffers", rad=0.0)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[완료] 소켓 서버 내부 파이프라인 다이어그램 생성: {output_filename}")

if __name__ == "__main__":
    generate_socket_server_pipeline()