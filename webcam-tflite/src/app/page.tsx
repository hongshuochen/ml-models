'use client';

import { useMemo, useState } from 'react';
import { CameraStage } from '@/components/CameraStage';
import { Controls } from '@/components/Controls';
import { ErrorBanner } from '@/components/ErrorBanner';
import { ModelSelector } from '@/components/ModelSelector';
import { ResultsPanel } from '@/components/ResultsPanel';
import { useCamera } from '@/hooks/useCamera';
import { useInference } from '@/hooks/useInference';
import { MODELS, getModelById } from '@/lib/models';

export default function Home() {
  const camera = useCamera();
  const [modelId, setModelId] = useState(MODELS[0].id);
  const [intervalMs, setIntervalMs] = useState(150);

  const model = useMemo(() => getModelById(modelId) ?? null, [modelId]);
  const cameraReady = camera.status === 'ready';

  // Inference runs continuously while the camera is live.
  const { modelStatus, modelError, result, inferError, stats, reload } = useInference(
    camera.videoRef,
    { model, running: cameraReady, intervalMs },
  );

  const cameraError =
    camera.status !== 'ready' && camera.status !== 'requesting' && camera.status !== 'idle';
  const showDet = result?.modelType === 'detection' || result?.modelType === 'twostage';
  const showPose = result?.modelType === 'pose' || result?.modelType === 'twostage';
  const detections = showDet ? result?.detections : undefined;
  const poses = showPose ? result?.poses : undefined;

  return (
    <main className="stage-root">
      <CameraStage videoRef={camera.videoRef} detections={detections} poses={poses} />

      <header className="hud hud-top">
        <div className="brand">
          <span className="live-dot" data-on={cameraReady} />
          Webcam · YOLO
        </div>
        <ModelSelector value={modelId} onChange={setModelId} disabled={modelStatus === 'loading'} />
      </header>

      <div className="hud hud-errors">
        <ErrorBanner
          message={cameraError ? camera.message : ''}
          onRetry={camera.status === 'unsupported' ? undefined : () => camera.start()}
        />
        <ErrorBanner
          message={
            modelStatus === 'error'
              ? `Model failed to load: ${modelError}`
              : modelStatus === 'unsupported'
                ? modelError
                : ''
          }
          onRetry={modelStatus === 'error' ? reload : undefined}
        />
      </div>

      <ResultsPanel
        result={result}
        stats={stats}
        inferError={inferError}
        modelName={modelStatus === 'loading' ? 'loading…' : (model?.shortName ?? '')}
      />

      <Controls
        intervalMs={intervalMs}
        onIntervalChange={setIntervalMs}
        devices={camera.devices}
        activeDeviceId={camera.activeDeviceId}
        onDeviceChange={(id) => camera.start(id)}
        disabled={!cameraReady}
      />
    </main>
  );
}
