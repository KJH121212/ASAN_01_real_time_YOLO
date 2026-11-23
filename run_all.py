"""ASAN Socket Server + Streamlit Client 단일 터미널 통합 실행기.

Ctrl+C 한 번으로 소켓 서버와 Streamlit 클라이언트(및 ffmpeg/adb 등 자식 프로세스)를 함께 종료한다.
"""

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT_DIR = Path(__file__).resolve().parent
SERVER_DIR = ROOT_DIR / "socket_server"
CLIENT_DIR = ROOT_DIR / "streamlit_client"


def _find_conda_rtm_pythons() -> list[Path]:
    home = Path.home()
    roots = [
        Path("C:/miniconda3"),
        home / "miniconda3",
        home / "anaconda3",
        Path("C:/anaconda3"),
        Path("C:/ProgramData/miniconda3"),
        Path("C:/ProgramData/anaconda3"),
    ]
    return [r / "envs" / "rtm_env_50" / "python.exe" for r in roots]


def _resolve_server_python() -> str:
    candidates = [
        SERVER_DIR / ".venv" / "Scripts" / "python.exe",
        SERVER_DIR / ".venv_onnx" / "Scripts" / "python.exe",
        *_find_conda_rtm_pythons(),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return sys.executable


def _resolve_client_python() -> str:
    candidates = [
        CLIENT_DIR / ".venv" / "Scripts" / "python.exe",
        *_find_conda_rtm_pythons(),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return sys.executable


def _kill_tree(proc: subprocess.Popen | None, label: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return
        except Exception:
            pass
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    print(f"[LAUNCHER] {label} stopped.", flush=True)


def _stream_output(proc: subprocess.Popen, tag: str, stop_event: threading.Event) -> None:
    if proc.stdout is None:
        return
    try:
        for line in proc.stdout:
            if stop_event.is_set():
                break
            text = line.rstrip("\r\n")
            if text:
                print(f"[{tag}] {text}", flush=True)
    except OSError:
        pass


def main() -> int:
    server_py = _resolve_server_python()
    client_py = _resolve_client_python()

    print("==================================================", flush=True)
    print("  ASAN Socket Server + Streamlit Client (Unified)", flush=True)
    print(f"  Server Python : {server_py}", flush=True)
    print(f"  Client Python : {client_py}", flush=True)
    print("  종료하려면 Ctrl+C 를 한 번 누르세요.", flush=True)
    print("==================================================", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    stop_event = threading.Event()
    server_proc: subprocess.Popen | None = None
    client_proc: subprocess.Popen | None = None

    def _handle_signal(_signum, _frame):
        stop_event.set()

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handle_signal)
            except Exception:
                pass

    try:
        server_proc = subprocess.Popen(
            [server_py, "-u", "server_entry.py"],
            cwd=str(SERVER_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        t_server = threading.Thread(target=_stream_output, args=(server_proc, "SERVER", stop_event), daemon=True)
        t_server.start()

        # 소켓 서버가 포트 바인딩을 시작할 수 있도록 잠시 대기
        for _ in range(15):
            if stop_event.is_set() or server_proc.poll() is not None:
                break
            time.sleep(0.1)

        if not stop_event.is_set() and server_proc.poll() is None:
            client_proc = subprocess.Popen(
                [client_py, "-m", "streamlit", "run", "app.py"],
                cwd=str(CLIENT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            t_client = threading.Thread(target=_stream_output, args=(client_proc, "CLIENT", stop_event), daemon=True)
            t_client.start()

        while not stop_event.is_set():
            if server_proc.poll() is not None:
                print(f"[LAUNCHER] Socket Server exited (code={server_proc.returncode}). Shutting down...", flush=True)
                break
            if client_proc is not None and client_proc.poll() is not None:
                print(f"[LAUNCHER] Streamlit Client exited (code={client_proc.returncode}). Shutting down...", flush=True)
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        print("\n[LAUNCHER] Stopping Socket Server and Streamlit Client...", flush=True)
        _kill_tree(client_proc, "Streamlit Client")
        _kill_tree(server_proc, "Socket Server")
        print("[LAUNCHER] All processes stopped cleanly.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
