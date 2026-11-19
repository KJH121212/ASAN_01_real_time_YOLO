"""소켓 서버 실험 화면. 실행: streamlit run app.py"""

import asyncio
import base64
from collections import deque
import copy
import inspect
import json
from pathlib import Path
import queue
import threading
import time

import streamlit as st
import websockets

from network.camera import configure_local_and_gopro_no_proxy, scan_available_cameras
from network.session_commands import available_output_modes

configure_local_and_gopro_no_proxy()
_WS_SUPPORTS_PROXY_KW = 'proxy' in inspect.signature(websockets.connect).parameters


def _ws_connect(uri: str, **kwargs):
    """websockets 15/16+에서 기본 활성화된 시스템 HTTP 프록시 터널링을 우회하여 로컬 서버에 직접 연결합니다."""
    if _WS_SUPPORTS_PROXY_KW:
        kwargs.setdefault('proxy', None)
    return websockets.connect(uri, **kwargs)


FINISH_EVENTS = {'SESSION_FINISHED', 'CALIBRATION_FINISHED', 'VIDEO_ANALYSIS_COMPLETED'}


class SocketExperiment:
    """화면 재실행 사이에도 하나의 측정 연결을 유지합니다."""

    def __init__(self, uri, packet):
        self.uri = uri
        self.packet = packet
        self.commands = queue.Queue(maxsize=8)
        self.lock = threading.Lock()
        self.thread = None
        self.preview_thread = None
        self.preview_closed = threading.Event()
        self.heartbeat = time.monotonic()
        self.stopping = False
        self.state = {'status': 'CONNECTING', 'pose': {}, 'server': {}, 'frame': None, 'progress': {},
                      'last_results': {},
                      'log': {}, 'summary': None, 'error': None, 'preview_warning': '', 'preview_notice': '',
                      'frame_received_at': None}
        self.events = deque(maxlen=30)

    def start(self):
        if self.packet.get('output_mode') == 'VISUAL':
            from network.video_window import video_window
            video_window.show(self)
            self.packet['preview_separate'] = True
            self.preview_thread = threading.Thread(target=lambda: asyncio.run(self._preview()), daemon=True)
            self.preview_thread.start()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    async def _preview(self):
        while not self.preview_closed.is_set():
            with self.lock:
                if time.monotonic()-self.heartbeat > 30:
                    return
            try:
                async with _ws_connect(self.uri, open_timeout=5, close_timeout=1,
                                       max_size=16*1024*1024, compression=None) as socket:
                    await socket.send(json.dumps({'type': 'CMD_WATCH_CAMERA',
                                                  'camera_index': self.packet.get('camera_index', 0),
                                                  'camera_max_width': self.packet.get('camera_max_width', 1280),
                                                  'camera_max_height': self.packet.get('camera_max_height', 720),
                                                  'exercise_name': self.packet.get('exercise_name', '')}))
                    while not self.preview_closed.is_set():
                        with self.lock:
                            if time.monotonic()-self.heartbeat > 30:
                                return
                        try:
                            packet = json.loads(await asyncio.wait_for(socket.recv(), timeout=.5))
                        except asyncio.TimeoutError:
                            continue
                        if packet.get('type') not in ('VIDEO_FRAME', 'PREVIEW_STATUS'):
                            continue
                        with self.lock:
                            if packet.get('type') == 'VIDEO_FRAME':
                                self.state['frame'] = base64.b64decode(packet['image_base64'], validate=True)
                                self.state['frame_received_at'] = time.monotonic()
                            self.state['preview_warning'] = packet.get('warning', '')
                            self.state['preview_notice'] = packet.get('notice', '')
            except Exception as exc:
                with self.lock:
                    self.state['preview_warning'] = f'영상 연결을 복구하고 있습니다: {exc}'
                await asyncio.sleep(1.)

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def snapshot(self):
        with self.lock:
            self.heartbeat = time.monotonic()
            return copy.deepcopy({**self.state, 'events': list(self.events)})

    def stop(self, reason='USER_STOP'):
        with self.lock:
            if self.stopping or not self.running:
                return
            self.stopping = True
            self.state['status'] = 'STOPPING'
        self.commands.put_nowait({'type': 'CMD_STOP_SESSION', 'reason': reason})

    def _worker(self):
        try:
            asyncio.run(self._communicate())
        except Exception as exc:
            with self.lock:
                self.state.update(status='ERROR', error=str(exc))

    async def _communicate(self):
        async with _ws_connect(self.uri, open_timeout=5, close_timeout=2,
                               max_size=16 * 1024 * 1024, compression=None) as socket:
            await socket.send(json.dumps(self.packet, allow_nan=False))
            with self.lock:
                self.state['status'] = 'PREPARING'
            stop_deadline = None
            while True:
                # 브라우저가 사라지면 서버가 측정을 계속하지 않도록 정상 종료를 요청합니다.
                with self.lock:
                    abandoned = time.monotonic() - self.heartbeat > 30
                if abandoned:
                    self.stop()
                try:
                    command = self.commands.get_nowait()
                except queue.Empty:
                    command = None
                if command:
                    await socket.send(json.dumps(command))
                    stop_deadline = time.monotonic() + 10
                if stop_deadline and time.monotonic() > stop_deadline:
                    raise TimeoutError('종료 응답이 없습니다. 로그 저장 여부를 서버에서 확인하세요.')
                try:
                    packet = json.loads(await asyncio.wait_for(socket.recv(), timeout=0.1))
                except asyncio.TimeoutError:
                    continue
                kind = packet.get('type')
                with self.lock:
                    if kind == 'VIDEO_FRAME':
                        self.state['frame'] = base64.b64decode(packet['image_base64'], validate=True)
                    elif kind == 'POSE_UPDATE':
                        self.state['pose'] = packet
                        if not self.stopping:
                            self.state['status'] = 'RUNNING'
                    elif kind == 'REP_COMPLETED':
                        side = packet.get('side')
                        if side in ('left', 'right'):
                            self.state['pose'].setdefault(side, {}).update(
                                rep_count=packet['rep_id'], quality=packet.get('quality'),
                                total_score=packet.get('total_score'), val=packet.get('val'))
                            self.state['pose']['mode'] = packet.get('mode', self.packet.get('mode', ''))
                            self.state['pose']['metric_type'] = packet.get('metric_type', self.state['pose'].get('metric_type', ''))
                            self.state['last_results'][side] = packet
                        self.events.appendleft(packet)
                    else:
                        self.events.appendleft(packet)
                    if kind == 'SERVER_STATUS':
                        self.state['server'] = packet
                    elif kind == 'SESSION_LOG_STARTED':
                        self.state['log'] = packet
                        self.state['pose'].update(mode=packet.get('mode', self.packet.get('mode', '')),
                            measurement=packet.get('measurement', {}).get('measurement', {}),
                            metric_type=packet.get('measurement', {}).get('metric_type', ''))
                    elif kind == 'VIDEO_PROGRESS':
                        self.state['progress'] = packet
                    elif kind == 'SESSION_PREPARING':
                        rem = packet.get('remaining_seconds')
                        if rem is not None:
                            self.state['preview_notice'] = f'측정 준비 중입니다 ({rem}초). 기본 자세를 취해 주세요.'
                    elif kind == 'SESSION_MEASUREMENT_STARTED':
                        self.state['preview_notice'] = ''
                        if not self.stopping:
                            self.state['status'] = 'RUNNING'
                    elif kind in {'SESSION_MEASUREMENT_STARTED', 'VIDEO_STATUS'} and not self.stopping:
                        self.state['status'] = 'RUNNING'
                    elif kind in FINISH_EVENTS:
                        self.state.update(status='FINISHED', summary=packet.get('summary'))
                    elif kind == 'ERROR':
                        self.state.update(status='ERROR', error=packet.get('message', '서버 오류'))
                    target_reached = (self.packet.get('mode') == 'MAIN'
                        and self.packet.get('auto_stop_on_target', False)
                        and all(self.state['pose'].get(side, {}).get('rep_count', 0) >= self.packet['target_reps']
                                for side in ('left', 'right')))
                if kind in {'POSE_UPDATE', 'REP_COMPLETED'} and target_reached:
                    self.stop(reason='TARGET_REACHED')
                if kind in FINISH_EVENTS or kind == 'ERROR':
                    return


async def check_server(uri):
    async with _ws_connect(uri, open_timeout=3, close_timeout=1) as socket:
        await socket.send(json.dumps({'type': 'CMD_GET_SERVER_STATUS'}))
        for _ in range(10):
            packet = json.loads(await asyncio.wait_for(socket.recv(), timeout=3))
            if packet.get('type') == 'SERVER_STATUS':
                return packet
        raise RuntimeError('서버 상태 응답을 받지 못했습니다.')


def measurement_tab():
    from network.streamlit_live import measurement_values

    st.caption('소켓 서버를 실행한 뒤 측정을 시작하세요. 로그와 720p 영상은 서버 PC에 저장됩니다.')
    client = st.session_state.get('experiment')
    active = client is not None and client.running
    if client is not None:
        if client.packet.get('output_mode') == 'VISUAL':
            if st.button('영상 창 다시 열기'):
                from network.video_window import video_window
                video_window.show(client)
            st.caption('측정 시작 시 영상 전용 창이 자동으로 열립니다. Esc 또는 창의 X 버튼으로 닫을 수 있습니다.')
    config = json.loads((Path(__file__).parent / 'configs/exercise_kpt_config_coco.json').read_text(encoding='utf-8'))
    exercises = config['exercises']
    with st.container(border=True):
        output_mode = st.radio('실행 방식', available_output_modes(), horizontal=True,
                               format_func=lambda value: {'VISUAL': '① 실시간 영상', 'LIGHTWEIGHT': '② 실시간 수치',
                                                          'OFFLINE_VIDEO': '③ 영상 파일 분석'}[value], disabled=active)
        if output_mode == 'OFFLINE_VIDEO':
            st.markdown("##### 📁 영상 파일 스켈레톤(NPZ) 추출")
            st.caption("비디오 파일에서 COCO 2D 관절 포인트와 측정 신호(Relative Y 등)를 추출하여 NPZ로 저장합니다. (횟수 카운트는 상단 '로그 분석' 탭에서 시뮬레이션)")

            c_ex, c_pat = st.columns([2, 1])
            exercise_keys = list(exercises)
            def_idx = exercise_keys.index('knee_extension') if 'knee_extension' in exercise_keys else 0
            exercise = c_ex.selectbox('운동 동작 선택', exercise_keys,
                                      format_func=lambda key: exercises[key].get('display_name_kr', key),
                                      index=def_idx, key='offline_exercise')
            patient = c_pat.text_input('환자번호', value='0000', max_chars=4, key='offline_patient')

            col_dir, col_b = st.columns([3, 1])
            default_dir = 'data/test_video/261109/knee_extension'
            if not Path(default_dir).is_dir():
                default_dir = 'data/test_video'
            video_folder = col_dir.text_input('영상 탐색 폴더', value=default_dir, key='video_search_dir',
                                              help='영상 파일이 위치한 폴더 경로입니다.')
            batch_size = col_b.number_input('배치 크기 (Batch Size)', min_value=1, value=16, step=1, key='offline_batch_size')

            v_path = Path(video_folder.strip().strip('"'))
            video_files = []
            if v_path.is_dir():
                video_files = sorted([
                    p for p in v_path.glob('*')
                    if p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv') and not p.name.startswith('.')
                ])
                if not video_files:
                    video_files = sorted([
                        p for p in v_path.glob('*/*')
                        if p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv') and not p.name.startswith('.')
                    ])

            video = ''
            if video_files:
                options = [str(p) for p in video_files] + ['직접 경로 입력...']
                def _fmt_video(p):
                    if p != '직접 경로 입력...':
                        try:
                            po = Path(p)
                            if po.is_file():
                                return f"{po.name} ({po.stat().st_size / (1024*1024):.1f} MB)"
                        except Exception:
                            pass
                    return p

                selected_opt = st.selectbox(
                    f'분석할 영상 파일 선택 (총 {len(video_files)}개 발견)',
                    options,
                    format_func=_fmt_video,
                    key='offline_selected_video'
                )
                if selected_opt == '직접 경로 입력...':
                    video = st.text_input('직접 영상 경로 입력', value='', key='offline_custom_video')
                else:
                    video = selected_opt
            else:
                video = st.text_input('영상 파일 경로 입력', value='', placeholder='예: data/test_video/261109/knee_extension/calibration.mp4', key='offline_video_input')

            from analysis.extract_video_npz import detect_mode_round_set, process_video_to_npz
            auto_mode, auto_round, auto_set = detect_mode_round_set(Path(video)) if video and Path(video).is_file() else ('CALIBRATION', 1, 1)

            st.caption(f"📌 저장 방식: `data/{patient.zfill(4)}/YYYY-MM-DD/{exercise}/` 아래 기존 정규 로그 형식 (`{auto_mode}`, Round {auto_round}, Set {auto_set})으로 저장됩니다.")

            if st.button('🚀 스켈레톤(NPZ) 추출 시작', type='primary', use_container_width=True, key='btn_offline_extract'):
                if not video or not Path(video).is_file():
                    st.error(f'올바른 영상 파일을 선택하거나 입력해 주세요: {video}')
                elif not patient.isascii() or not patient.isdigit():
                    st.error('환자번호는 숫자 1~4자리로 입력하세요.')
                else:
                    bar_container = st.progress(0.0, text="추출 준비 중...")
                    def on_progress(cur, total, speed, msg):
                        pct = min(1.0, max(0.0, cur / total)) if total > 0 else 0.0
                        bar_container.progress(pct, text=f"[{pct*100:.1f}%] {msg}")

                    try:
                        npz_p, csv_p = process_video_to_npz(
                            video_path=Path(video),
                            exercise_name=exercise,
                            patient_id=patient.zfill(4),
                            mode=auto_mode,
                            round_number=auto_round,
                            set_number=auto_set,
                            batch_size=int(batch_size),
                            on_progress=on_progress,
                        )
                        bar_container.progress(1.0, text="[100.0%] 추출 및 기존 포맷 저장 완료!")
                        st.session_state['log_paths'] = [str(npz_p.resolve())]
                        st.session_state['last_extracted_npz'] = str(npz_p.resolve())
                        st.session_state['last_extracted_video'] = str(Path(video).resolve())
                        st.success(f"🎉 기존 정규 규격으로 저장 완료!\n- **NPZ**: `{npz_p}`\n- **CSV**: `{csv_p}`")
                        st.info("💡 상단의 **[로그 분석]** 탭으로 이동하시면 추출된 관절 신호 확인 및 **[카운트 실험]**을 진행하실 수 있습니다.")
                    except Exception as exc:
                        st.error(f"추출 실패: {exc}")

        else:
            left, right = st.columns(2)
            patient = left.text_input('환자번호', value='0000', max_chars=4, disabled=active)
            mode = right.selectbox('측정 모드', ['CALIBRATION', 'MAIN'],
                                   format_func=lambda value: {'CALIBRATION': '캘리브레이션', 'MAIN': '본 측정'}[value],
                                   disabled=active)
            exercise = st.selectbox('동작', list(exercises),
                                    format_func=lambda key: exercises[key].get('display_name_kr', key), disabled=active)
            patient_name = st.text_input('환자 이름 (선택)', disabled=active)
            round_col, set_col, target_col = st.columns(3)
            round_number = round_col.number_input('라운드', min_value=1, value=1, step=1, disabled=active)
            set_number = set_col.number_input('세트', min_value=1, value=1, step=1, disabled=active)
            target_reps = target_col.number_input('목표 횟수 (좌우 각각)', min_value=1,
                value=3 if mode == 'CALIBRATION' else 10, step=1, key=f'target_reps_{mode}',
                disabled=active or mode == 'CALIBRATION', help='캘리브레이션은 좌우 3회 고정입니다.')
            auto_stop = st.checkbox('양쪽 모두 목표 횟수에 도달하면 종료 · 저장', value=True,
                                    disabled=active or mode == 'CALIBRATION')
            cam_col1, cam_col2, cam_col3 = st.columns([1.4, 1.8, 1.0])
            camera = cam_col1.number_input('카메라 선택 (번호)', min_value=0, value=2, step=1, disabled=active,
                                           help='소켓 서버 PC의 카메라 번호입니다. 선택한 번호가 검은 화면/비활성이면 활성 카메라(예: 2번)로 자동 전환됩니다.')
            res_presets = {
                '720p (1280×720 · 권장 표준)': (1280, 720),
                '540p (960×540 · 고속/지연 최소)': (960, 540),
                '360p (640×360 · 초경량)': (640, 360),
                '1080p (1920×1080 · 고화질)': (1920, 1080),
            }
            res_choice = cam_col2.selectbox(
                '고프로 · 카메라 최대 입력 화질 (속도 최적화)',
                list(res_presets.keys()),
                index=0,
                disabled=active,
                help='고프로에서 1080p/4K 고화질 영상이 들어오더라도 선택한 해상도로 즉시 다운스케일하여 추론·전송 지연을 방지합니다.',
            )
            cam_max_w, cam_max_h = res_presets[res_choice]
            cam_col3.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if cam_col3.button('🔍 카메라 · 고프로 진단', disabled=active, use_container_width=True):
                with st.spinner('카메라 인덱스 및 GoPro USB / 프록시 상태 점검 중...'):
                    st.session_state['camera_diag'] = scan_available_cameras(max_index=4)

            diag = st.session_state.get('camera_diag')
            if diag:
                cams = diag.get('cameras', [])
                cam_text = (
                    ' / '.join(
                        f"**{c['index']}번 [{c.get('name', 'Camera')}]** ({c.get('status_tag', '')} · {c['resolution']})"
                        for c in cams
                    )
                    if cams else '감지된 카메라 없음'
                )
                gopro_ip = diag.get('gopro_usb_ip')
                gopro_msg = f"감지됨 (`{gopro_ip}`)" if gopro_ip else "미감지 (일반 웹캠 또는 DirectShow 가상 드라이버 모드)"
                proxies = diag.get('system_proxies_detected') or {}
                proxy_msg = (
                    f"시스템 프록시 감지됨 ({proxies}) → **자동 우회(`proxy=None`, `NO_PROXY`) 적용 완료**"
                    if proxies else "시스템 프록시 없음 (직접 연결)"
                )
                st.info(
                    f"📷 **사용 가능한 카메라**: {cam_text}\n\n"
                    f"📡 **GoPro USB 이더넷**: {gopro_msg}\n\n"
                    f"🛡️ **프록시 상태**: {proxy_msg}"
                )
            st.caption('카메라를 바꾸려면 측정을 종료한 뒤 번호를 변경하세요. 번호와 실제 장치의 대응은 PC마다 다릅니다.')
            with st.expander('연결 · 입력 설정'):
                uri = st.text_input('소켓 서버 주소', value='ws://127.0.0.1:8080', disabled=active)
                preparation = st.number_input('측정 준비 시간 (초)', min_value=0., value=5., step=.5, disabled=active)
                log_root = st.text_input('로그 저장 폴더 (서버 PC, 비우면 기본 data)', disabled=active)
                session_col, registration_col, slot_col = st.columns(3)
                session_number = session_col.number_input('세션 번호', min_value=1, value=1, step=1, disabled=active)
                registration = registration_col.number_input('등록 차수', min_value=1, value=1, step=1, disabled=active)
                slot = slot_col.number_input('슬롯', min_value=1, value=1, step=1, disabled=active)
                data_version = st.text_input('데이터 버전', value='v1.0.0(0)', disabled=active)
                target_score = st.number_input('참고 목표 점수', min_value=.25, value=10., step=.25, disabled=active,
                    help='서버의 품질 점수 기준입니다. 자동 종료는 위의 목표 횟수로 판단하며 점수와 구분합니다.')
                play_id = st.text_input('play_session_id (선택, 비우면 자동)', disabled=active)
                record_id = st.text_input('exercise_record_id (선택, 비우면 자동)', disabled=active)
                metadata_text = st.text_area('환자 부가 정보 JSON (선택)', value='{}', disabled=active,
                                             help='예: {"height_cm": 170, "weight_kg": 65}')
            st.caption(f'설정: {round_number}라운드 · {set_number}세트 · 좌우 각 {target_reps}회. '
                       + ('캘리브레이션은 바의 30% 도달 즉시 +1, 15% 이하 복귀 후 다시 셉니다.' if mode == 'CALIBRATION'
                          else '양쪽 목표 도달 시 자동 저장합니다.' if auto_stop else '종료 · 저장 버튼으로 마칩니다.'))
            if mode == 'MAIN':
                st.caption('같은 환자·날짜·동작·라운드의 최신 확정 캘리브레이션을 사용합니다. 기준이 없으면 캘리브레이션부터 진행하세요.')
            start_col, stop_col, check_col = st.columns(3)
            with start_col:
                start_requested = st.button('측정 시작', type='primary', disabled=active, use_container_width=True)
            if start_requested:
                if not patient.isascii() or not patient.isdigit():
                    st.error('환자번호는 숫자 1~4자리로 입력하세요.')
                elif not uri.startswith(('ws://', 'wss://')):
                    st.error('서버 주소는 ws:// 또는 wss://로 시작해야 합니다.')
                else:
                    packet = {'type': 'CMD_SET_SESSION', 'patient_id': patient.zfill(4),
                              'patient_name': patient_name.strip() or 'Unknown',
                              'exercise_name': exercise, 'mode': mode, 'output_mode': output_mode,
                              'session': int(session_number), 'round': int(round_number), 'set': int(set_number),
                              'target_reps': int(target_reps), 'target_score': float(target_score),
                              'auto_stop_on_target': bool(auto_stop), 'preparation_seconds': float(preparation),
                              'registration': int(registration), 'slot': int(slot), 'data_version': data_version.strip(),
                              'camera_index': int(camera),
                              'camera_max_width': int(cam_max_w), 'camera_max_height': int(cam_max_h)}
                    try:
                        metadata = json.loads(metadata_text.strip() or '{}')
                        if not isinstance(metadata, dict):
                            raise ValueError('환자 부가 정보는 JSON 객체여야 합니다.')
                        json.dumps(metadata, allow_nan=False)
                    except ValueError as exc:
                        st.error(f'환자 부가 정보를 확인하세요: {exc}')
                        return
                    if metadata:
                        packet['patient_metadata'] = metadata
                    for name, value in (('log_root', log_root), ('play_session_id', play_id), ('exercise_record_id', record_id)):
                        if value.strip():
                            packet[name] = value.strip().strip('"')
                    previous = client
                    if previous is not None:
                        client.preview_closed.set()
                    client = SocketExperiment(uri.strip(), packet)
                    if previous is not None and output_mode == 'VISUAL':
                        snapshot = previous.snapshot()
                        client.state['frame'] = snapshot['frame']
                        client.state['frame_received_at'] = snapshot.get('frame_received_at')
                    st.session_state['experiment'] = client
                    client.start()
                    st.rerun()
            if stop_col.button('종료 · 저장', disabled=not active, use_container_width=True):
                client.stop()
            if check_col.button('연결 확인', disabled=active, use_container_width=True):
                try:
                    st.session_state['server_check'] = asyncio.run(check_server(uri.strip()))
                except Exception as exc:
                    st.session_state['server_check'] = {'message': str(exc), 'model_state': 'ERROR'}

    @st.fragment(run_every=0.2)
    def results():
        current = st.session_state.get('experiment')
        if current is None:
            checked = st.session_state.get('server_check')
            if checked:
                st.info(f"{checked.get('model_state')}: {checked.get('message')}")
            else:
                st.info('측정을 시작하면 진행 상태와 측정값이 여기에 표시됩니다.')
            return
        state = current.snapshot()
        if active != current.running:
            st.rerun()  # 종료되면 설정과 시작 버튼을 다시 활성화합니다.
        st.write(f"상태: **{state['status']}**")
        setting = current.packet
        st.caption(f"현재 측정: {setting.get('round', 1)}라운드 · {setting.get('set', 1)}세트 · "
                   f"목표 좌우 각 {setting.get('target_reps', 3 if setting['mode'] == 'CALIBRATION' else 10)}회")
        calibration = (state['summary'] or state['log']).get('calibration')
        if calibration:
            st.caption(f"캘리브레이션 기준: {calibration['file']} · {calibration['created_at']}")
        progress = state['progress']
        if progress:
            total = progress.get('total_frames', 0)
            processed = progress.get('processed_frames', 0)
            label = '관절 추출' if progress.get('phase') == 'EXTRACTING' else '동작 판정'
            if state['summary']:
                st.progress(1.0, text='분석 완료')
            elif total > 0:
                st.progress(min(1.0, processed / total), text=f'{label}: {processed}/{total} 프레임')
            else:
                st.caption(f'{label}: {processed} 프레임')
        if state['error']:
            st.error(state['error'])
        if state.get('preview_warning'):
            st.warning(state['preview_warning'])
        server = state['server']
        if server and not server.get('ready'):
            st.info(server.get('message', '모델 준비 중'))
        pose = state['pose']
        measurement_values(state)
        log = state['summary'] or state['log']
        if log.get('directory'):
            st.caption('저장 위치 (서버 PC)')
            st.code(log['directory'], language=None)
        if state['summary']:
            st.success('서버가 로그 저장을 완료했습니다.')
            video_path = state['summary'].get('video_mp4')
            if current.packet.get('output_mode') == 'OFFLINE_VIDEO' and video_path:
                with st.expander('분석한 원본 영상'):
                    if Path(video_path).is_file():
                        st.video(video_path)
                    else:
                        st.caption('영상은 소켓 서버 PC의 저장 경로에서 확인하세요.')
        with st.expander('수신 패킷 · 저장 결과'):
            st.json({'request': current.packet, 'pose': pose, 'summary': state['summary'], 'events': state['events']})

    results()


def main():
    st.set_page_config(page_title='모션 측정 실험실', page_icon='📡', layout='wide')
    from analysis.ui import show_analysis

    st.title('모션 측정 실험실')
    measurement, analysis = st.tabs(['측정', '로그 분석'])
    with measurement:
        measurement_tab()
    with analysis:
        show_analysis()


if __name__ == '__main__':
    main()
