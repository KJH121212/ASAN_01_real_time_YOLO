"""Instant interactive client-side Skeleton Motion Player with Synchronized Joint Graph.

Renders pure skeleton animation on HTML5 Canvas directly in the browser
synchronized with a real-time interactive time-series curve below it.
Features face-hiding, trajectory smoothing (One-Euro / rolling Gaussian),
and instant click-to-pose scrubbing.
"""

import json
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from analysis.log_reader import coordinates


def smooth_keypoints(points: np.ndarray, window: int = 5, max_gap: int = 3) -> np.ndarray:
    """Smooth keypoint coordinates over time while interpolating brief dropouts.

    Args:
        points: (N, 17, 3) array where [:, :, :2] are x, y and [:, :, 2] is confidence.
        window: Rolling window size (odd integer, e.g. 3, 5, 7).
        max_gap: Maximum missing frames to linearly interpolate.
    Returns:
        smoothed_points: (N, 17, 3) array.
    """
    if len(points) == 0:
        return points

    smoothed = points.copy()
    window = max(3, window if window % 2 == 1 else window + 1)

    for j in range(17):
        for ax in (0, 1):
            s = pd.Series(points[:, j, ax])
            # 1) Linearly fill small drops (up to max_gap frames)
            s_interp = s.interpolate(method='linear', limit=max_gap)
            # 2) Centered rolling mean to eliminate jitter
            s_smooth = s_interp.rolling(window=window, min_periods=1, center=True).mean()
            smoothed[:, j, ax] = s_smooth.to_numpy()

    return smoothed


def build_skeleton_player_html(
    points_nx17x3: np.ndarray,
    timestamps: List[float],
    is_norm: bool = True,
    is_smoothed: bool = False,
    hide_face: bool = True,
    aspect_ratio: float = 1.7778,
    fps: float = 30.0,
    width: int = 620,
    skeleton_height: int = 460,
    chart_height: int = 150,
    chart_series: Optional[List[Dict]] = None,
    chart_title: str = "선택 관절 시계열 그래프 (클릭하여 스켈레톤 확인)",
) -> str:
    """Generate self-contained HTML5 Canvas player + synchronized joint time-series chart."""
    total_frames = len(points_nx17x3)
    if total_frames == 0:
        return "<div style='color:white;padding:20px;'>프레임 데이터가 없습니다.</div>"

    # Convert coordinates to lightweight JSON
    frames_list = []
    for f in range(total_frames):
        frame_pts = []
        for j in range(17):
            x, y, s = points_nx17x3[f, j]
            if np.isfinite([x, y, s]).all():
                frame_pts.append([round(float(x), 4), round(float(y), 4), round(float(s), 2)])
            else:
                frame_pts.append(None)
        frames_list.append(frame_pts)

    times_list = [round(float(t), 3) for t in timestamps] if timestamps is not None and len(timestamps) == total_frames else [round(i / fps, 3) for i in range(total_frames)]

    # Clean chart series payload
    clean_series = []
    if chart_series:
        for s in chart_series:
            raw_vals = s.get("values", [])
            vals = [round(float(v), 4) if (v is not None and np.isfinite(v)) else None for v in raw_vals]
            clean_series.append({
                "name": str(s.get("name", "")),
                "color": str(s.get("color", "#38bdf8")),
                "values": vals,
                "dashed": bool(s.get("dashed", False))
            })

    payload_json = json.dumps({
        "frames": frames_list,
        "times": times_list,
        "chart_series": clean_series,
        "chart_title": chart_title
    }, ensure_ascii=False)

    norm_txt = "신체 정규화 (Norm)" if is_norm else "원본 영상 비율 (Unnorm)"
    smooth_txt = " · Smoothed" if is_smoothed else " · Raw"
    mode_label = f"{norm_txt}{smooth_txt}"

    html_code = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background: transparent; color: #f1f5f9; display: flex; flex-direction: column; align-items: center; justify-content: center; overflow: hidden; }}
  .player-card {{
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 12px;
    width: 100%;
    max-width: {width + 24}px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }}
  .canvas-container {{
    position: relative;
    width: 100%;
    max-width: {width}px;
    height: {skeleton_height}px;
    background: #050811;
    border-radius: 8px;
    border: 1px solid #334155;
    overflow: hidden;
  }}
  #skelCanvas {{
    width: 100%;
    height: 100%;
    display: block;
  }}
  .hud-overlay {{
    position: absolute;
    top: 10px;
    left: 12px;
    font-size: 13px;
    font-weight: 500;
    color: #e2e8f0;
    background: rgba(15, 23, 42, 0.75);
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.1);
    pointer-events: none;
    line-height: 1.4;
  }}
  .hud-mode {{
    position: absolute;
    top: 10px;
    right: 12px;
    font-size: 11px;
    font-weight: 600;
    color: #38bdf8;
    background: rgba(15, 23, 42, 0.85);
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid rgba(56, 189, 248, 0.25);
    pointer-events: none;
  }}
  .controls-bar {{
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .slider-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
  }}
  input[type=range] {{
    flex: 1;
    accent-color: #38bdf8;
    cursor: pointer;
    height: 6px;
    background: #334155;
    border-radius: 3px;
  }}
  .buttons-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    flex-wrap: wrap;
    gap: 6px;
  }}
  .btn-group {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  button {{
    background: #1e293b;
    color: #f1f5f9;
    border: 1px solid #475569;
    padding: 5px 11px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }}
  button:hover {{
    background: #334155;
    border-color: #64748b;
  }}
  button.active {{
    background: #0284c7;
    border-color: #38bdf8;
    color: #ffffff;
  }}
  .loop-toggle {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    color: #94a3b8;
    cursor: pointer;
    user-select: none;
  }}
  .loop-toggle input {{
    accent-color: #38bdf8;
    cursor: pointer;
  }}
  /* Chart section */
  .chart-section {{
    width: 100%;
    max-width: {width}px;
    background: #090e1a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 6px 8px 4px 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    position: relative;
  }}
  .chart-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 11px;
    color: #94a3b8;
  }}
  .chart-hint {{
    font-size: 10px;
    color: #38bdf8;
  }}
  #chartCanvas {{
    width: 100%;
    height: {chart_height}px;
    display: block;
    cursor: crosshair;
    border-radius: 6px;
  }}
</style>
</head>
<body>

<div class="player-card">
  <!-- 1. Skeleton Canvas Display -->
  <div class="canvas-container" id="canvasBox">
    <canvas id="skelCanvas" width="{width}" height="{skeleton_height}"></canvas>
    <div class="hud-overlay" id="hudInfo">프레임: 1 / {total_frames} · 0.00초</div>
    <div class="hud-mode">{mode_label}</div>
  </div>

  <!-- 2. Controls & Scrubber Slider -->
  <div class="controls-bar">
    <div class="slider-row">
      <input type="range" id="frameSlider" min="0" max="{total_frames - 1}" value="0">
    </div>

    <div class="buttons-row">
      <div class="btn-group">
        <button id="btnPlay">⏸️ 일시정지</button>
        <button id="btnPrev" title="이전 프레임 (←)">⏮️</button>
        <button id="btnNext" title="다음 프레임 (→)">⏭️</button>
        <button id="btnReset" title="처음으로">🔄</button>
      </div>

      <div class="btn-group">
        <span style="font-size:11px; color:#94a3b8; margin-right:2px;">속도:</span>
        <button class="btn-speed" data-speed="0.5">0.5x</button>
        <button class="btn-speed active" data-speed="1.0">1.0x</button>
        <button class="btn-speed" data-speed="1.5">1.5x</button>
        <button class="btn-speed" data-speed="2.0">2.0x</button>
      </div>

      <label class="loop-toggle" title="끝나면 처음부터 무한 재생 (GIF 모드)">
        <input type="checkbox" id="chkLoop" checked>
        <span>🔁 무한 반복 (GIF)</span>
      </label>
    </div>
  </div>

  <!-- 3. Synchronized Interactive Chart -->
  <div class="chart-section" id="chartBox">
    <div class="chart-header">
      <span id="chartTitleText">📈 {chart_title}</span>
      <span class="chart-hint">💡 그래프를 클릭/드래그하면 스켈레톤이 즉시 동기화됩니다</span>
    </div>
    <canvas id="chartCanvas" width="{width}" height="{chart_height}"></canvas>
  </div>
</div>

<script>
  const DATA = {payload_json};
  const frames = DATA.frames;
  const times = DATA.times;
  const seriesList = DATA.chart_series || [];
  const totalFrames = frames.length;
  const isNorm = {"true" if is_norm else "false"};
  const hideFace = {"true" if hide_face else "false"};
  const videoAspect = {aspect_ratio};
  const baseFps = {fps};

  const skelCanvas = document.getElementById('skelCanvas');
  const skelCtx = skelCanvas.getContext('2d');
  const chartCanvas = document.getElementById('chartCanvas');
  const chartCtx = chartCanvas ? chartCanvas.getContext('2d') : null;

  const hudInfo = document.getElementById('hudInfo');
  const slider = document.getElementById('frameSlider');
  const btnPlay = document.getElementById('btnPlay');
  const btnPrev = document.getElementById('btnPrev');
  const btnNext = document.getElementById('btnNext');
  const btnReset = document.getElementById('btnReset');
  const chkLoop = document.getElementById('chkLoop');
  const speedButtons = document.querySelectorAll('.btn-speed');

  let currentFrame = 0;
  let isPlaying = true;
  let playbackSpeed = 1.0;
  let lastTime = performance.now();
  let isDraggingChart = false;

  const ALL_EDGES = [
    [0, 1], [0, 2], [1, 3], [2, 4],          // Head
    [5, 6], [5, 11], [6, 12], [11, 12],      // Torso
    [5, 7], [7, 9],                           // Left Arm
    [6, 8], [8, 10],                          // Right Arm
    [11, 13], [13, 15],                       // Left Leg
    [12, 14], [14, 16]                        // Right Leg
  ];

  const BODY_EDGES = [
    [5, 6], [5, 11], [6, 12], [11, 12],      // Torso
    [5, 7], [7, 9],                           // Left Arm
    [6, 8], [8, 10],                          // Right Arm
    [11, 13], [13, 15],                       // Left Leg
    [12, 14], [14, 16]                        // Right Leg
  ];

  const EDGES = hideFace ? BODY_EDGES : ALL_EDGES;
  const startJointIdx = hideFace ? 5 : 0;

  function getEdgeColor(i, j) {{
    if ((i === 11 || i === 13 || i === 15) && (j === 11 || j === 13 || j === 15)) return '#eab308'; // Left leg (Yellow)
    if ((i === 12 || i === 14 || i === 16) && (j === 12 || j === 14 || j === 16)) return '#ef4444'; // Right leg (Red)
    if ((i === 5 || i === 7 || i === 9) && (j === 5 || j === 7 || j === 9)) return '#38bdf8';       // Left arm (Sky)
    if ((i === 6 || i === 8 || i === 10) && (j === 6 || j === 8 || j === 10)) return '#fb923c';     // Right arm (Orange)
    if ((i === 5 || i === 6 || i === 11 || i === 12) && (j === 5 || j === 6 || j === 11 || j === 12)) return '#10b981'; // Torso (Emerald)
    return '#94a3b8'; // Head
  }}

  function getNodeStyle(idx) {{
    if (idx === 15 || idx === 16) return {{ radius: 6.5, color: '#facc15' }}; // Ankles
    if (idx === 13 || idx === 14) return {{ radius: 6, color: '#fb923c' }};   // Knees
    if (idx === 11 || idx === 12) return {{ radius: 5.5, color: '#10b981' }}; // Hips
    if (idx === 5 || idx === 6)   return {{ radius: 5, color: '#38bdf8' }};   // Shoulders
    return {{ radius: 3.5, color: '#cbd5e1' }};
  }}

  // 1. Draw Skeleton (Optionally hides face joints 0~4)
  function drawSkeleton(frameIdx) {{
    const w = skelCanvas.width;
    const h = skelCanvas.height;
    skelCtx.clearRect(0, 0, w, h);

    const grad = skelCtx.createRadialGradient(w/2, h/2, 10, w/2, h/2, Math.max(w, h)/1.2);
    grad.addColorStop(0, '#0a101f');
    grad.addColorStop(1, '#020617');
    skelCtx.fillStyle = grad;
    skelCtx.fillRect(0, 0, w, h);

    const kpts = frames[frameIdx];
    if (!kpts || kpts.length === 0) return;

    let ptsMap = {{}};
    if (isNorm) {{
      const cx = w / 2;
      const cy = h / 2 - 15;
      const scale = h / 2.7;
      for (let i = startJointIdx; i < 17; i++) {{
        const p = kpts[i];
        if (p && p[0] !== null && p[1] !== null) {{
          ptsMap[i] = {{ x: cx + p[0] * scale, y: cy - p[1] * scale, s: p[2] }};
        }}
      }}
    }} else {{
      let drawW = w;
      let drawH = h;
      let offX = 0;
      let offY = 0;
      const canvasAspect = w / h;
      if (videoAspect > canvasAspect) {{
        drawW = w;
        drawH = w / videoAspect;
        offY = (h - drawH) / 2;
      }} else {{
        drawH = h;
        drawW = h * videoAspect;
        offX = (w - drawW) / 2;
      }}
      for (let i = startJointIdx; i < 17; i++) {{
        const p = kpts[i];
        if (p && p[0] !== null && p[1] !== null) {{
          ptsMap[i] = {{ x: offX + p[0] * drawW, y: offY + p[1] * drawH, s: p[2] }};
        }}
      }}
    }}

    // Draw bones
    skelCtx.lineWidth = 3.5;
    skelCtx.lineCap = 'round';
    for (const [i, j] of EDGES) {{
      const p1 = ptsMap[i];
      const p2 = ptsMap[j];
      if (p1 && p2) {{
        skelCtx.strokeStyle = getEdgeColor(i, j);
        skelCtx.beginPath();
        skelCtx.moveTo(p1.x, p1.y);
        skelCtx.lineTo(p2.x, p2.y);
        skelCtx.stroke();
      }}
    }}

    // Draw joint nodes (skips face joints 0~4 if hideFace is true)
    for (let i = startJointIdx; i < 17; i++) {{
      const p = ptsMap[i];
      if (p) {{
        const style = getNodeStyle(i);
        skelCtx.fillStyle = style.color;
        skelCtx.beginPath();
        skelCtx.arc(p.x, p.y, style.radius, 0, Math.PI * 2);
        skelCtx.fill();
        skelCtx.strokeStyle = '#000000';
        skelCtx.lineWidth = 1.5;
        skelCtx.stroke();
      }}
    }}

    // Update HUD
    const t = times[frameIdx] !== undefined ? times[frameIdx] : (frameIdx / baseFps);
    hudInfo.innerText = `프레임: ${{frameIdx + 1}} / ${{totalFrames}} · ${{t.toFixed(2)}}초`;
  }}

  // 2. Draw Synchronized Joint Chart
  function drawChart(frameIdx) {{
    if (!chartCtx || seriesList.length === 0) return;
    const w = chartCanvas.width;
    const h = chartCanvas.height;
    chartCtx.clearRect(0, 0, w, h);

    chartCtx.fillStyle = '#050811';
    chartCtx.fillRect(0, 0, w, h);

    const padL = 48;
    const padR = 16;
    const padT = 22;
    const padB = 20;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    let minVal = Infinity;
    let maxVal = -Infinity;
    for (const s of seriesList) {{
      for (const v of s.values) {{
        if (v !== null && !isNaN(v)) {{
          if (v < minVal) minVal = v;
          if (v > maxVal) maxVal = v;
        }}
      }}
    }}
    if (!isFinite(minVal) || !isFinite(maxVal) || minVal === maxVal) {{
      minVal = -1;
      maxVal = 1;
    }}
    const span = maxVal - minVal;
    minVal -= span * 0.08;
    maxVal += span * 0.08;

    const valToY = (v) => padT + (1 - (v - minVal) / (maxVal - minVal)) * plotH;
    const frameToX = (f) => padL + (f / Math.max(1, totalFrames - 1)) * plotW;

    // Horizontal grid lines
    chartCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    chartCtx.lineWidth = 1;
    chartCtx.setLineDash([3, 3]);

    const yTicks = [minVal, (minVal + maxVal) / 2, maxVal];
    chartCtx.fillStyle = '#64748b';
    chartCtx.font = '10px monospace';
    chartCtx.textAlign = 'right';

    for (let i = 0; i < 3; i++) {{
      const v = yTicks[i];
      const y = valToY(v);
      chartCtx.beginPath();
      chartCtx.moveTo(padL, y);
      chartCtx.lineTo(padL + plotW, y);
      chartCtx.stroke();
      chartCtx.fillText(v.toFixed(2), padL - 6, y + 3);
    }}
    chartCtx.setLineDash([]);

    // Draw curve lines (dashed supported for Raw curves)
    for (const s of seriesList) {{
      chartCtx.strokeStyle = s.color;
      chartCtx.lineWidth = s.dashed ? 1.4 : 2.0;
      if (s.dashed) {{
        chartCtx.setLineDash([4, 3]);
      }} else {{
        chartCtx.setLineDash([]);
      }}
      chartCtx.beginPath();
      let started = false;
      for (let f = 0; f < totalFrames; f++) {{
        const v = s.values[f];
        if (v !== null && !isNaN(v)) {{
          const x = frameToX(f);
          const y = valToY(v);
          if (!started) {{
            chartCtx.moveTo(x, y);
            started = true;
          }} else {{
            chartCtx.lineTo(x, y);
          }}
        }} else {{
          started = false;
        }}
      }}
      chartCtx.stroke();
      chartCtx.setLineDash([]);
    }}

    // Vertical playback scrubber line
    const curX = frameToX(frameIdx);
    chartCtx.strokeStyle = '#f43f5e';
    chartCtx.lineWidth = 1.5;
    chartCtx.setLineDash([4, 2]);
    chartCtx.beginPath();
    chartCtx.moveTo(curX, padT);
    chartCtx.lineTo(curX, padT + plotH);
    chartCtx.stroke();
    chartCtx.setLineDash([]);

    // Interactive Legend & Points at current frame
    let legendX = padL;
    chartCtx.textAlign = 'left';
    chartCtx.font = '11px sans-serif';

    for (const s of seriesList) {{
      const curV = s.values[frameIdx];
      chartCtx.fillStyle = s.color;
      chartCtx.beginPath();
      chartCtx.arc(legendX + 4, 11, 3.5, 0, Math.PI * 2);
      chartCtx.fill();

      const label = (curV !== null && !isNaN(curV))
        ? `${{s.name}}: ${{curV.toFixed(3)}}`
        : `${{s.name}}: -`;
      chartCtx.fillStyle = '#e2e8f0';
      chartCtx.fillText(label, legendX + 12, 15);
      legendX += chartCtx.measureText(label).width + 18;

      if (curV !== null && !isNaN(curV) && !s.dashed) {{
        const py = valToY(curV);
        chartCtx.fillStyle = s.color;
        chartCtx.beginPath();
        chartCtx.arc(curX, py, 4.5, 0, Math.PI * 2);
        chartCtx.fill();
        chartCtx.strokeStyle = '#ffffff';
        chartCtx.lineWidth = 1.5;
        chartCtx.stroke();
      }}
    }}

    // Time boundaries
    chartCtx.fillStyle = '#64748b';
    chartCtx.font = '10px sans-serif';
    chartCtx.textAlign = 'left';
    chartCtx.fillText('0.0초', padL, h - 4);
    chartCtx.textAlign = 'right';
    const totalT = times.length > 0 ? times[totalFrames - 1] : (totalFrames - 1) / baseFps;
    chartCtx.fillText(`${{totalT.toFixed(1)}}초`, padL + plotW, h - 4);
  }}

  function updatePlayBtn() {{
    btnPlay.innerText = isPlaying ? '⏸️ 일시정지' : '▶️ 재생';
  }}

  function renderBoth(idx) {{
    drawSkeleton(idx);
    drawChart(idx);
  }}

  function loop(now) {{
    if (isPlaying) {{
      const interval = (1000 / baseFps) / playbackSpeed;
      const elapsed = now - lastTime;
      if (elapsed >= interval) {{
        currentFrame++;
        if (currentFrame >= totalFrames) {{
          if (chkLoop.checked) {{
            currentFrame = 0;
          }} else {{
            currentFrame = totalFrames - 1;
            isPlaying = false;
            updatePlayBtn();
          }}
        }}
        slider.value = currentFrame;
        renderBoth(currentFrame);
        lastTime = now - (elapsed % interval);
      }}
    }}
    requestAnimationFrame(loop);
  }}

  // Chart Click / Drag Scrubber
  function updateFrameFromChartMouse(e) {{
    const rect = chartCanvas.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const padL = 48;
    const padR = 16;
    const plotW = chartCanvas.width - padL - padR;
    const scaleX = chartCanvas.width / rect.width;
    const xInCanvas = clientX * scaleX;

    const clampedX = Math.max(padL, Math.min(padL + plotW, xInCanvas));
    const pct = (clampedX - padL) / plotW;
    const targetFrame = Math.round(pct * (totalFrames - 1));

    currentFrame = Math.max(0, Math.min(totalFrames - 1, targetFrame));
    slider.value = currentFrame;
    renderBoth(currentFrame);
  }}

  if (chartCanvas) {{
    chartCanvas.addEventListener('mousedown', (e) => {{
      isDraggingChart = true;
      isPlaying = false; // pause to allow precise posture inspection
      updatePlayBtn();
      updateFrameFromChartMouse(e);
    }});

    window.addEventListener('mousemove', (e) => {{
      if (isDraggingChart) {{
        updateFrameFromChartMouse(e);
      }}
    }});

    window.addEventListener('mouseup', () => {{
      isDraggingChart = false;
    }});
  }}

  // Button Controls
  btnPlay.onclick = () => {{
    isPlaying = !isPlaying;
    updatePlayBtn();
    lastTime = performance.now();
  }};

  btnReset.onclick = () => {{
    currentFrame = 0;
    slider.value = 0;
    renderBoth(0);
    lastTime = performance.now();
  }};

  btnPrev.onclick = () => {{
    isPlaying = false;
    updatePlayBtn();
    currentFrame = Math.max(0, currentFrame - 1);
    slider.value = currentFrame;
    renderBoth(currentFrame);
  }};

  btnNext.onclick = () => {{
    isPlaying = false;
    updatePlayBtn();
    currentFrame = Math.min(totalFrames - 1, currentFrame + 1);
    slider.value = currentFrame;
    renderBoth(currentFrame);
  }};

  slider.oninput = (e) => {{
    currentFrame = parseInt(e.target.value, 10);
    renderBoth(currentFrame);
  }};

  speedButtons.forEach(btn => {{
    btn.onclick = () => {{
      speedButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      playbackSpeed = parseFloat(btn.getAttribute('data-speed'));
    }};
  }});

  // Keyboard navigation
  window.addEventListener('keydown', (e) => {{
    if (e.code === 'Space') {{
      e.preventDefault();
      btnPlay.click();
    }} else if (e.code === 'ArrowLeft') {{
      e.preventDefault();
      btnPrev.click();
    }} else if (e.code === 'ArrowRight') {{
      e.preventDefault();
      btnNext.click();
    }}
  }});

  // Start immediately
  renderBoth(0);
  requestAnimationFrame(loop);
</script>

</body>
</html>
"""
    return html_code


def render_skeleton_player_component(
    log: dict,
    is_norm: bool = True,
    confidence: float = 0.35,
    apply_smoothing: bool = True,
    smooth_window: int = 5,
    hide_face: bool = True,
    chart_series: Optional[List[Dict]] = None,
    chart_title: str = "선택 관절 시계열 그래프 (클릭 시 스켈레톤 동기화)",
    height: int = 770,
):
    """Embed the interactive HTML5 canvas skeleton player + synchronized chart into Streamlit."""
    mode = "신체 정규화" if is_norm else "영상 비율"
    pts = coordinates(log, mode=mode, confidence=confidence)

    # Apply smoothing if requested
    if apply_smoothing:
        pts = smooth_keypoints(pts, window=smooth_window)

    times = log.get("timestamps", [])
    if isinstance(times, np.ndarray):
        times = times.tolist()

    aspects = log.get("image_aspect_ratio", 1.7778)
    if isinstance(aspects, (np.ndarray, list)) and len(aspects) > 0:
        valid_aspects = [float(a) for a in aspects if np.isfinite(a) and float(a) > 0]
        aspect = valid_aspects[0] if valid_aspects else 1.7778
    else:
        aspect = float(aspects) if np.isfinite(aspects) else 1.7778

    html_code = build_skeleton_player_html(
        points_nx17x3=pts,
        timestamps=times,
        is_norm=is_norm,
        is_smoothed=apply_smoothing,
        hide_face=hide_face,
        aspect_ratio=aspect,
        fps=30.0,
        width=620,
        skeleton_height=460,
        chart_height=150,
        chart_series=chart_series,
        chart_title=chart_title,
    )
    components.html(html_code, height=height)
