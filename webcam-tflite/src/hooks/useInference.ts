'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { ModelDef, ResultMsg, WorkerToMain } from '@/lib/types';

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'error' | 'unsupported';

interface Options {
  model: ModelDef | null;
  running: boolean;
  intervalMs: number;
}

export interface InferenceStats {
  latencyMs: number; // model inference time
  fps: number; // achieved inference rate
}

/**
 * Owns the inference Web Worker and the frame-capture loop. Frames are captured
 * on the main thread (cheap canvas resize), but inference runs in the worker, so
 * the UI never blocks. Backpressure: at most one frame is in flight at a time.
 * Results are tagged with a model id so results from a previous model selection
 * are dropped after a switch.
 */
export function useInference(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  { model, running, intervalMs }: Options,
) {
  const [modelStatus, setModelStatus] = useState<ModelStatus>('idle');
  const [modelError, setModelError] = useState('');
  const [result, setResult] = useState<ResultMsg | null>(null);
  const [inferError, setInferError] = useState('');
  const [stats, setStats] = useState<InferenceStats>({ latencyMs: 0, fps: 0 });

  const workerRef = useRef<Worker | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const busyRef = useRef(false);
  const reqIdRef = useRef(0);
  const lastSentRef = useRef(0);
  const lastResultAtRef = useRef(0);

  // Loop-relevant values kept in refs so the rAF loop need not be recreated.
  const modelRef = useRef<ModelDef | null>(model);
  const intervalRef = useRef(intervalMs);
  useEffect(() => {
    modelRef.current = model;
    intervalRef.current = intervalMs;
  }, [model, intervalMs]);

  // Create the worker once.
  useEffect(() => {
    if (typeof Worker === 'undefined') {
      setModelStatus('unsupported');
      setModelError('Web Workers are not supported in this browser.');
      return;
    }
    const worker = new Worker('/workers/inference.worker.js');
    workerRef.current = worker;
    canvasRef.current = document.createElement('canvas');

    worker.onmessage = (e: MessageEvent<WorkerToMain>) => {
      const msg = e.data;
      const currentId = modelRef.current?.id;
      // Ignore anything that belongs to a previously selected model.
      if (msg.modelId !== currentId) {
        if (msg.type === 'result' || msg.type === 'inferError') busyRef.current = false;
        return;
      }
      switch (msg.type) {
        case 'ready':
          setModelStatus('ready');
          break;
        case 'loadError':
          setModelStatus('error');
          setModelError(msg.error);
          break;
        case 'result': {
          busyRef.current = false;
          setResult(msg);
          setInferError('');
          const now = performance.now();
          const dt = now - lastResultAtRef.current;
          lastResultAtRef.current = now;
          setStats({
            latencyMs: Math.round(msg.inferenceMs),
            fps: dt > 0 && dt < 5000 ? Math.round((1000 / dt) * 10) / 10 : 0,
          });
          break;
        }
        case 'inferError':
          busyRef.current = false;
          setInferError(msg.error);
          break;
      }
    };
    worker.onerror = (e) => {
      busyRef.current = false; // never wedge the capture loop on a worker fault
      setModelStatus('error');
      setModelError(e.message || 'Worker crashed.');
    };

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  const loadModel = useCallback((m: ModelDef) => {
    const worker = workerRef.current;
    if (!worker) return;
    busyRef.current = false;
    setResult(null);
    setInferError('');
    setModelError('');
    setModelStatus('loading');
    worker.postMessage({ type: 'load', model: m });
  }, []);

  // (Re)load whenever the selection changes.
  useEffect(() => {
    if (model) loadModel(model);
  }, [model, loadModel]);

  /** Retry loading the current model (e.g. after a fetch failure or crash). */
  const reload = useCallback(() => {
    if (modelRef.current) loadModel(modelRef.current);
  }, [loadModel]);

  // Frame-capture loop — only scheduled while actively running on a ready model,
  // so it does not wake the main thread when paused/idle.
  useEffect(() => {
    if (!running || modelStatus !== 'ready') return;
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const worker = workerRef.current;
      const video = videoRef.current;
      const m = modelRef.current;
      if (!worker || !video || !m || busyRef.current || video.readyState < 2 || video.videoWidth === 0) {
        return;
      }
      const now = performance.now();
      if (now - lastSentRef.current < intervalRef.current) return;
      lastSentRef.current = now;

      const canvas = canvasRef.current!;
      if (canvas.width !== m.inputWidth || canvas.height !== m.inputHeight) {
        canvas.width = m.inputWidth;
        canvas.height = m.inputHeight;
        ctxRef.current = canvas.getContext('2d', { willReadFrequently: true });
      }
      const ctx = ctxRef.current;
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
      busyRef.current = true;
      reqIdRef.current += 1;
      worker.postMessage(
        {
          type: 'infer',
          requestId: reqIdRef.current,
          modelId: m.id,
          width: canvas.width,
          height: canvas.height,
          buffer: img.data.buffer,
        },
        [img.data.buffer],
      );
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [videoRef, running, modelStatus]);

  return { modelStatus, modelError, result, inferError, stats, reload };
}
