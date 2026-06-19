'use client';

import type { InferenceStats } from '@/hooks/useInference';
import type { ResultMsg } from '@/lib/types';

interface Props {
  result: ResultMsg | null;
  stats: InferenceStats;
  inferError: string;
  modelName: string;
}

export function ResultsPanel({ result, stats, inferError, modelName }: Props) {
  const isPose = result?.modelType === 'pose';
  const count = (isPose ? result?.poses?.length : result?.detections?.length) ?? 0;
  const unit = isPose ? (count === 1 ? 'hand' : 'hands') : count === 1 ? 'face' : 'faces';
  const visibleKpts = isPose
    ? (result?.poses ?? []).reduce((n, p) => n + p.keypoints.filter((k) => k.score > 0).length, 0)
    : 0;

  return (
    <div className="results glass hud">
      <div className="results-head">
        <span className="results-title">{modelName}</span>
        <span className="results-stat">
          {stats.latencyMs} ms · {stats.fps} fps
        </span>
      </div>

      {inferError ? (
        <p className="err-text">{inferError}</p>
      ) : !result ? (
        <p className="muted">warming up…</p>
      ) : (
        <>
          <div className="count">
            <span className="count-num">{count}</span>
            <span className="count-unit">{unit}</span>
          </div>
          {isPose && count > 0 && <div className="sub-line">{visibleKpts} keypoints tracked</div>}
        </>
      )}
    </div>
  );
}
