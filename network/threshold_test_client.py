"""Interactive Unity-style client: start webcam tracking, inspect/change ROM."""

import argparse
import asyncio
import json
import threading
import uuid

import websockets


HELP = "Commands: show | left START TARGET | right START TARGET | both START TARGET | quit"


def make_command(text, exercise_name):
    words = text.strip().split()
    if words == ["show"]:
        packet = {"type": "CMD_GET_COUNT_THRESHOLD"}
    elif len(words) == 3 and words[0] in {"left", "right", "both"}:
        start, target = float(words[1]), float(words[2])
        sides = ("left", "right") if words[0] == "both" else (words[0],)
        packet = {
            "type": "CMD_UPDATE_COUNT_THRESHOLD",
            "thresholds": {side: {"start_val": start, "target_val": target} for side in sides},
        }
    else:
        raise ValueError(HELP)
    packet.update(exercise_name=exercise_name, request_id=uuid.uuid4().hex)
    return packet


async def run(args):
    # Keep session start, updates and streaming on the same WebSocket.
    async with websockets.connect(args.uri) as ws:
        await ws.send(json.dumps({
            "type": "CMD_SET_SESSION",
            "player_id": args.player_id,
            "patient_name": "Threshold Test",
            "exercise_name": args.exercise_name,
            "mode": "MAIN",
            "target_score": args.target_score,
            "output_mode": "LIGHTWEIGHT",
            "camera_index": args.camera_index,
        }))
        await ws.send(json.dumps(make_command("show", args.exercise_name)))
        print(HELP, flush=True)
        print("Values use the active metric (degrees or normalized height/distance).", flush=True)

        commands = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def read_console():
            while True:
                try:
                    text = input("\nthreshold> ")
                except (EOFError, KeyboardInterrupt):
                    text = "quit"
                try:
                    loop.call_soon_threadsafe(commands.put_nowait, text)
                except RuntimeError:
                    return
                if text.strip().lower() == "quit":
                    return

        async def send_commands():
            while True:
                text = (await commands.get()).strip()
                if text.lower() == "quit":
                    return  # Closing this connection saves the session and closes the camera.
                if text.lower() == "help" or not text:
                    print(HELP, flush=True)
                    continue
                try:
                    packet = make_command(text, args.exercise_name)
                    await ws.send(json.dumps(packet, allow_nan=False))
                except ValueError as exc:
                    print(f"Invalid command: {exc}", flush=True)

        async def receive():
            previous = None
            async for message in ws:
                packet = json.loads(message)
                if packet.get("type") == "POSE_UPDATE":
                    left, right = packet.get("left", {}), packet.get("right", {})
                    status = (left.get("rep_count"), right.get("rep_count"),
                              left.get("state"), right.get("state"),
                              packet.get("is_occluded"), packet.get("threshold_version"))
                    if status != previous:
                        print(f"\nL={left.get('rep_count')} ({left.get('state')}, {left.get('quality')}) "
                              f"R={right.get('rep_count')} ({right.get('state')}, {right.get('quality')}) "
                              f"occluded={packet.get('is_occluded')} "
                              f"ROM version={packet.get('threshold_version')}", flush=True)
                        previous = status
                else:
                    print("\n" + json.dumps(packet, ensure_ascii=False, indent=2), flush=True)
                    if packet.get("type") in {"SESSION_FINISHED", "CALIBRATION_FINISHED"}:
                        return

        threading.Thread(target=read_console, daemon=True).start()
        tasks = {asyncio.create_task(send_commands()), asyncio.create_task(receive())}
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="ws://127.0.0.1:8080")
    parser.add_argument("--exercise-name", default="biceps_curl")
    parser.add_argument("--player-id", default="threshold_test")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--target-score", "--target-reps", dest="target_score", type=float, default=1000,
                        help="Target points (Perfect=1, Good=0.5, Bad=0.25)")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
    except (OSError, websockets.exceptions.ConnectionClosed) as exc:
        print(f"Connection ended: {exc}")


if __name__ == "__main__":
    main()
