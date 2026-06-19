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
  const type = result?.modelType;
  const faces = result?.detections?.length ?? 0;
  const hands = result?.poses?.length ?? 0;

  let body = null;
  if (result) {
    if (type === 'twostage') {
      body = (
        <div className="count">
          <span className="count-num">{faces}</span>
          <span className="count-unit">{faces === 1 ? 'face' : 'faces'}</span>
          <span className="count-num" style={{ marginLeft: 14 }}>{hands}</span>
          <span className="count-unit">{hands === 1 ? 'hand' : 'hands'}</span>
        </div>
      );
    } else if (type === 'pose') {
      body = (
        <div className="count">
          <span className="count-num">{hands}</span>
          <span className="count-unit">{hands === 1 ? 'hand' : 'hands'}</span>
        </div>
      );
    } else {
      body = (
        <div className="count">
          <span className="count-num">{faces}</span>
          <span className="count-unit">{faces === 1 ? 'face' : 'faces'}</span>
        </div>
      );
    }
  }

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
        body
      )}
    </div>
  );
}
