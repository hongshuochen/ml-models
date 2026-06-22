// Shared types for the app and the inference Web Worker protocol.

export type ModelType = 'detection' | 'pose' | 'twostage';

/** A bundled YOLO TFLite model and everything needed to run/interpret it. */
export interface ModelDef {
  id: string;
  name: string;
  /** Short label for the segmented selector. */
  shortName: string;
  type: ModelType;
  /** Path under /public to the .tflite file (exported from Ultralytics, NMS baked in). */
  modelUrl: string;
  inputWidth: number;
  inputHeight: number;
  /** Class names indexed by the model's class id. */
  classNames: string[];
  /** Two-stage only: landmark regressor run on each detected hand crop. */
  landmarkUrl?: string;
  landmarkInput?: number;
  /** Landmark input layout fallback if shape can't be read: 'nchw' (default) or 'nhwc'. */
  landmarkLayout?: 'nchw' | 'nhwc';
  /** Minimum confidence to keep a detection/pose. */
  scoreThreshold: number;
  /** Keypoint score threshold (pose models only). */
  keypointThreshold?: number;
  description: string;
}

// ---- Results ---------------------------------------------------------------

/** Box coords normalized to [0,1] relative to the full (square) model input. */
export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Detection {
  label: string;
  score: number; // 0..1
  box: Box;
}

export interface Keypoint {
  x: number; // normalized 0..1
  y: number; // normalized 0..1
  score: number; // 0..1
}

export interface Pose {
  label: string;
  score: number;
  box: Box;
  keypoints: Keypoint[];
}

// ---- Worker message protocol ----------------------------------------------

export interface LoadMsg {
  type: 'load';
  model: ModelDef;
}

export interface InferMsg {
  type: 'infer';
  requestId: number;
  /** Id of the model this frame is meant for (used to drop stale results). */
  modelId: string;
  width: number;
  height: number;
  /** RGBA pixel buffer of size width*height*4 (transferred). */
  buffer: ArrayBuffer;
}

export type MainToWorker = LoadMsg | InferMsg;

export interface ReadyMsg {
  type: 'ready';
  modelId: string;
}

export interface LoadErrorMsg {
  type: 'loadError';
  modelId: string;
  error: string;
}

export interface ResultMsg {
  type: 'result';
  requestId: number;
  modelId: string;
  modelType: ModelType;
  inferenceMs: number;
  detections?: Detection[];
  poses?: Pose[];
}

export interface InferErrorMsg {
  type: 'inferError';
  requestId: number;
  modelId: string;
  error: string;
}

export type WorkerToMain = ReadyMsg | LoadErrorMsg | ResultMsg | InferErrorMsg;
