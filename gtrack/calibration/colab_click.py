# -*- coding: utf-8 -*-
"""
Colab 안에서 영상 프레임을 클릭해 픽셀 좌표를 받는 도구 (원본 도구의 cv2 창 + 확대경 대체)

사용
  pts = click_points(frame_bgr, n_points=4, labels=CORNER_LABELS)

조작
  1) 큰 화면에서 대략 클릭 → 오른쪽 확대경이 그 위치에 고정된다 (주황 십자 = 후보점).
  2) 확대경 안을 클릭하면 후보점을 정밀하게 옮긴다 (확대 배율만큼 정밀).
     키보드 방향키로 0.25 px씩 미세 이동도 된다 (화면을 한 번 클릭한 뒤).
  3) [점 확정] 또는 Enter/Space로 확정 → 다음 점으로 넘어간다.
  4) 모두 찍으면 [완료]. [되돌리기]는 마지막 확정점 취소, [초기화]는 전체 삭제.
픽셀 좌표는 항상 원본 해상도 기준이다.
"""

import base64
import json
import uuid

import cv2
import numpy as np

_HTML = r"""
<div id="__UID___wrap" style="font-family:sans-serif;font-size:14px;">
  <div id="__UID___info" style="margin:4px 0 6px 0;font-weight:bold;"></div>
  <div style="display:flex;gap:12px;align-items:flex-start;">
    <canvas id="__UID___cv" style="max-width:__MAXW__px;width:100%;border:1px solid #888;cursor:crosshair;"></canvas>
    <div>
      <canvas id="__UID___lens" width="__LENS__" height="__LENS__"
              style="width:__LENS__px;height:__LENS__px;border:2px solid #444;cursor:crosshair;"></canvas>
      <div style="margin-top:6px;display:flex;flex-direction:column;gap:4px;">
        <button id="__UID___ok">점 확정 (Enter)</button>
        <button id="__UID___undo">되돌리기</button>
        <button id="__UID___clear">초기화</button>
        <button id="__UID___done" style="font-weight:bold;">완료</button>
      </div>
      <div id="__UID___list" style="margin-top:6px;font-family:monospace;font-size:12px;white-space:pre;"></div>
    </div>
  </div>
  <img id="__UID___img" style="display:none;" src="data:image/jpeg;base64,__B64__">
</div>
"""

_JS = r"""
(function() {
  const uid = '__UID__';
  const N = __N__;
  const labels = __LABELS__;
  const ZOOM = __ZOOM__, LENS = __LENS__;
  const img = document.getElementById(uid + '_img');
  const cv = document.getElementById(uid + '_cv');
  const lens = document.getElementById(uid + '_lens');
  const info = document.getElementById(uid + '_info');
  const list = document.getElementById(uid + '_list');
  const ctx = cv.getContext('2d');
  const lctx = lens.getContext('2d');
  lctx.imageSmoothingEnabled = false;
  let pts = [];
  let pending = null;   // 확정 전 후보점 [x, y]
  let hover = null;     // 마우스 위치 [x, y]
  let finished = false;
  let resolver = null;

  function scale() { return cv.width / cv.getBoundingClientRect().width; }
  function toImg(e) {
    const r = cv.getBoundingClientRect();
    return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height];
  }
  function mark(c, x, y, rad, color, text) {
    c.beginPath(); c.arc(x, y, rad, 0, 2 * Math.PI); c.fillStyle = color; c.fill();
    if (text) {
      c.font = (rad * 3) + 'px sans-serif'; c.lineWidth = rad / 2; c.strokeStyle = 'black';
      c.strokeText(text, x + rad * 2, y - rad * 2); c.fillStyle = color; c.fillText(text, x + rad * 2, y - rad * 2);
    }
  }
  function cross(c, x, y, len, color, lw) {
    c.strokeStyle = color; c.lineWidth = lw; c.beginPath();
    c.moveTo(x - len, y); c.lineTo(x + len, y); c.moveTo(x, y - len); c.lineTo(x, y + len); c.stroke();
  }
  function drawMain() {
    ctx.drawImage(img, 0, 0);
    const s = scale(), rad = 5 * s;
    if (pts.length > 1) {
      ctx.strokeStyle = 'yellow'; ctx.lineWidth = 1.5 * s; ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      if (pts.length === N && N > 2) ctx.closePath();
      ctx.stroke();
    }
    pts.forEach((p, i) => mark(ctx, p[0], p[1], rad, 'yellow', String(i + 1)));
    if (pending) cross(ctx, pending[0], pending[1], 14 * s, 'orange', 2 * s);
  }
  function drawLens() {
    const c = pending || hover;
    lctx.fillStyle = '#222'; lctx.fillRect(0, 0, LENS, LENS);
    if (!c) return;
    const src = LENS / ZOOM;
    lctx.drawImage(img, c[0] - src / 2, c[1] - src / 2, src, src, 0, 0, LENS, LENS);
    pts.forEach((p, i) => {
      const lx = (p[0] - c[0]) * ZOOM + LENS / 2, ly = (p[1] - c[1]) * ZOOM + LENS / 2;
      if (lx >= 0 && lx <= LENS && ly >= 0 && ly <= LENS) mark(lctx, lx, ly, 4, 'yellow', String(i + 1));
    });
    cross(lctx, LENS / 2, LENS / 2, LENS / 2, pending ? 'orange' : 'lime', 1);
  }
  function refresh() {
    drawMain(); drawLens();
    const next = pts.length < N ? labels[pts.length] : '';
    info.textContent = finished ? '완료되었습니다. 다음 셀을 실행하세요.'
      : (pts.length < N ? `${pts.length}/${N} 확정 — 다음: ${next}` + (pending ? '  (후보점 있음: 확대경에서 조정 후 [점 확정])' : '')
                        : `${N}점 모두 확정 — [완료]를 누르세요`);
    list.textContent = pts.map((p, i) => `${i + 1}: (${p[0].toFixed(2)}, ${p[1].toFixed(2)})`).join('\n')
      + (pending ? `\n후보: (${pending[0].toFixed(2)}, ${pending[1].toFixed(2)})` : '');
  }
  function confirmPending() {
    if (pending && pts.length < N) { pts.push(pending); pending = null; refresh(); }
  }

  cv.addEventListener('mousemove', e => { hover = toImg(e); if (!pending) drawLens(); });
  cv.addEventListener('click', e => { if (finished || pts.length >= N) return; pending = toImg(e); refresh(); });
  lens.addEventListener('click', e => {
    const c = pending || hover;
    if (finished || !c || pts.length >= N) return;
    const r = lens.getBoundingClientRect();
    const lx = (e.clientX - r.left) * LENS / r.width, ly = (e.clientY - r.top) * LENS / r.height;
    pending = [c[0] + (lx - LENS / 2) / ZOOM, c[1] + (ly - LENS / 2) / ZOOM];
    refresh();
  });
  document.addEventListener('keydown', e => {
    if (finished) return;
    if ((e.key === 'Enter' || e.key === ' ') && pending) { confirmPending(); e.preventDefault(); return; }
    if (!pending) return;
    const step = 0.25;
    if (e.key === 'ArrowLeft') pending[0] -= step;
    else if (e.key === 'ArrowRight') pending[0] += step;
    else if (e.key === 'ArrowUp') pending[1] -= step;
    else if (e.key === 'ArrowDown') pending[1] += step;
    else return;
    e.preventDefault(); refresh();
  });
  document.getElementById(uid + '_ok').onclick = confirmPending;
  document.getElementById(uid + '_undo').onclick = () => { if (finished) return; if (pending) pending = null; else pts.pop(); refresh(); };
  document.getElementById(uid + '_clear').onclick = () => { if (finished) return; pts = []; pending = null; refresh(); };
  document.getElementById(uid + '_done').onclick = () => {
    if (pts.length !== N) { info.textContent = `아직 ${pts.length}/${N}점입니다.`; return; }
    finished = true; refresh();
    if (resolver) resolver(JSON.stringify(pts));
  };
  window[uid + '_wait'] = () => new Promise(res => {
    resolver = res;
    if (finished) res(JSON.stringify(pts));
  });

  function init() { cv.width = img.naturalWidth; cv.height = img.naturalHeight; refresh(); }
  if (img.complete && img.naturalWidth) init(); else img.onload = init;
})();
"""


def click_points(image_bgr, n_points=4, labels=None, max_display_width=1100,
                 zoom=6, lens_size=260, jpeg_quality=95):
    """Colab 출력 칸에 클릭 화면을 띄우고, [완료]를 누르면 원본 픽셀 좌표 (n,2) 배열을 반환한다."""
    from IPython.display import display, HTML, Javascript
    from google.colab import output

    if labels is None:
        labels = [f'{i + 1}번 점' for i in range(n_points)]
    ok, buf = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if not ok:
        raise RuntimeError('이미지 인코딩 실패')
    uid = 'gt' + uuid.uuid4().hex[:10]
    html = (_HTML.replace('__UID__', uid)
            .replace('__B64__', base64.b64encode(buf.tobytes()).decode('ascii'))
            .replace('__MAXW__', str(int(max_display_width)))
            .replace('__LENS__', str(int(lens_size))))
    js = (_JS.replace('__UID__', uid)
          .replace('__N__', str(int(n_points)))
          .replace('__LABELS__', json.dumps(list(labels), ensure_ascii=False))
          .replace('__ZOOM__', str(float(zoom)))
          .replace('__LENS__', str(int(lens_size))))
    display(HTML(html))
    display(Javascript(js))
    result = output.eval_js(f'window.{uid}_wait()')
    pts = np.array(json.loads(result), dtype=np.float32).reshape(-1, 2)
    print('클릭 좌표(원본 픽셀):', np.round(pts, 2).tolist())
    return pts
