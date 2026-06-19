'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

export type CameraStatus =
  | 'idle'
  | 'requesting'
  | 'ready'
  | 'denied'
  | 'notfound'
  | 'inuse'
  | 'unsupported'
  | 'error';

export interface CameraState {
  status: CameraStatus;
  message: string;
  devices: MediaDeviceInfo[];
  activeDeviceId: string | null;
}

const STATUS_MESSAGE: Record<CameraStatus, string> = {
  idle: '',
  requesting: 'Requesting camera access…',
  ready: '',
  denied: 'Camera permission was denied. Allow camera access in your browser and retry.',
  notfound: 'No camera device was found on this system.',
  inuse: 'The camera is already in use by another application.',
  unsupported:
    'This browser does not support camera capture (getUserMedia), or the page is not served over a secure context (https / localhost).',
  error: 'Could not start the camera.',
};

const ENDED_MESSAGE =
  'Camera stream ended (device disconnected or permission revoked). Click Retry to reconnect.';

function mapError(err: unknown): CameraStatus {
  switch ((err as DOMException)?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'denied';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'notfound';
    case 'NotReadableError':
    case 'AbortError':
      return 'inuse';
    default:
      return 'error';
  }
}

/** Manages getUserMedia lifecycle, device enumeration, and friendly errors. */
export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const trackRef = useRef<MediaStreamTrack | null>(null);
  const onEndedRef = useRef<(() => void) | null>(null);
  // Serializes overlapping start() calls (StrictMode double-mount, rapid device
  // switches) so a superseded request disposes its stream instead of leaking it.
  const reqIdRef = useRef(0);

  const [state, setState] = useState<CameraState>({
    status: 'idle',
    message: '',
    devices: [],
    activeDeviceId: null,
  });

  const stop = useCallback(() => {
    if (trackRef.current && onEndedRef.current) {
      trackRef.current.removeEventListener('ended', onEndedRef.current);
    }
    trackRef.current = null;
    onEndedRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const start = useCallback(
    async (deviceId?: string) => {
      const supported =
        typeof navigator !== 'undefined' &&
        !!navigator.mediaDevices?.getUserMedia &&
        (typeof window === 'undefined' || window.isSecureContext);
      if (!supported) {
        setState((s) => ({ ...s, status: 'unsupported', message: STATUS_MESSAGE.unsupported }));
        return;
      }

      const myReq = ++reqIdRef.current;
      setState((s) => ({ ...s, status: 'requesting', message: STATUS_MESSAGE.requesting }));
      stop();
      try {
        const constraints: MediaStreamConstraints = {
          audio: false,
          video: deviceId
            ? { deviceId: { exact: deviceId } }
            : { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (myReq !== reqIdRef.current) {
          // A newer start() superseded us — release this stream, don't commit.
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }

        // Surface involuntary teardown (unplug / OS revoke) as a retryable error.
        const track = stream.getVideoTracks()[0] ?? null;
        if (track) {
          const onEnded = () => {
            if (trackRef.current === track) {
              setState((s) => ({ ...s, status: 'error', message: ENDED_MESSAGE }));
            }
          };
          track.addEventListener('ended', onEnded);
          trackRef.current = track;
          onEndedRef.current = onEnded;
        }

        let devices: MediaDeviceInfo[] = [];
        try {
          devices = (await navigator.mediaDevices.enumerateDevices()).filter(
            (d) => d.kind === 'videoinput',
          );
        } catch {
          /* keep going with an empty device list */
        }
        if (myReq !== reqIdRef.current) return;
        const activeId = track?.getSettings().deviceId ?? deviceId ?? null;
        setState({ status: 'ready', message: '', devices, activeDeviceId: activeId });
      } catch (err) {
        if (myReq !== reqIdRef.current) return; // a newer start owns the camera now
        stop();
        const status = mapError(err);
        setState((s) => ({ ...s, status, message: STATUS_MESSAGE[status] }));
      }
    },
    [stop],
  );

  // Auto-request on mount; keep the device list fresh; clean up on unmount.
  useEffect(() => {
    start();
    const md = typeof navigator !== 'undefined' ? navigator.mediaDevices : undefined;
    const onDeviceChange = async () => {
      try {
        const devices = (await md!.enumerateDevices()).filter((d) => d.kind === 'videoinput');
        setState((s) => (s.status === 'ready' ? { ...s, devices } : s));
      } catch {
        /* ignore */
      }
    };
    md?.addEventListener?.('devicechange', onDeviceChange);
    return () => {
      md?.removeEventListener?.('devicechange', onDeviceChange);
      stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { videoRef, ...state, start, stop };
}
