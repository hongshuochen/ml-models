'use client';

import { useEffect, useRef } from 'react';
import type { Detection, Pose } from '@/lib/types';

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  detections?: Detection[];
  poses?: Pose[];
}

const BOX_COLORS = ['#22d3ee', '#a3e635', '#f472b6', '#fbbf24', '#60a5fa', '#f87171'];

// 21-point hand topology (MediaPipe / Ultralytics hand-keypoints order).
const HAND_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/**
 * Full-screen <video> (object-fit: cover) with an overlay canvas. The overlay
 * maps normalized model coordinates onto the same cover-fit rect the browser
 * uses to paint the video, so boxes/keypoints stay aligned at any aspect ratio.
 */
export function CameraStage({ videoRef, detections, poses }: Props) {
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ew = video.clientWidth;
    const eh = video.clientHeight;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!ew || !eh || !vw || !vh) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = ew * dpr;
    canvas.height = eh * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, ew, eh);

    // object-fit: cover mapping
    const scale = Math.max(ew / vw, eh / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    const ox = (ew - dw) / 2;
    const oy = (eh - dh) / 2;
    const px = (nx: number) => ox + nx * dw;
    const py = (ny: number) => oy + ny * dh;

    if (detections?.length) {
      ctx.lineWidth = 2.5;
      ctx.font = '600 13px ui-sans-serif, system-ui, sans-serif';
      ctx.textBaseline = 'top';
      detections.forEach((d, i) => {
        const color = BOX_COLORS[i % BOX_COLORS.length];
        const x = px(d.box.x);
        const y = py(d.box.y);
        const bw = d.box.width * dw;
        const bh = d.box.height * dh;
        ctx.strokeStyle = color;
        ctx.strokeRect(x, y, bw, bh);
        const text = `${d.label} ${(d.score * 100).toFixed(0)}%`;
        const tw = ctx.measureText(text).width;
        const ty = y > 18 ? y - 18 : y;
        ctx.fillStyle = color;
        ctx.fillRect(x, ty, tw + 8, 18);
        ctx.fillStyle = '#0b0f17';
        ctx.fillText(text, x + 4, ty + 2);
      });
    }

    if (poses?.length) {
      poses.forEach((p) => {
        ctx.strokeStyle = 'rgba(34,211,238,0.5)';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(px(p.box.x), py(p.box.y), p.box.width * dw, p.box.height * dh);
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 3;
        for (const [a, b] of HAND_EDGES) {
          const ka = p.keypoints[a];
          const kb = p.keypoints[b];
          if (!ka || !kb || ka.score === 0 || kb.score === 0) continue;
          ctx.beginPath();
          ctx.moveTo(px(ka.x), py(ka.y));
          ctx.lineTo(px(kb.x), py(kb.y));
          ctx.stroke();
        }
        ctx.fillStyle = '#fbbf24';
        for (const k of p.keypoints) {
          if (k.score === 0) continue;
          ctx.beginPath();
          ctx.arc(px(k.x), py(k.y), 3.5, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    }
  }, [detections, poses, videoRef]);

  return (
    <>
      <video ref={videoRef} className="cam-video" playsInline muted autoPlay />
      <canvas ref={overlayRef} className="cam-overlay" />
    </>
  );
}
