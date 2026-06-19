import type { ModelDef } from './types';

/**
 * Bundled YOLO26-nano models, trained locally and exported to TFLite with NMS
 * baked in (Ultralytics `yolo export format=tflite nms=True`). Files live under
 * /public/models. Outputs are fixed-size [1, 300, N] with normalized coords:
 *   detection: N=6  -> [x1, y1, x2, y2, conf, cls]
 *   pose:      N=69 -> [x1, y1, x2, y2, conf, cls, 21 x (kx, ky, kconf)]
 */
export const MODELS: ModelDef[] = [
  {
    id: 'face-hand-2stage',
    name: 'Face + Hand (2-stage landmarks)',
    shortName: 'Face+Hand',
    type: 'twostage',
    modelUrl: '/models/face_hand_yolo26n.tflite',
    landmarkUrl: '/models/hand_landmark.tflite',
    landmarkInput: 224,
    inputWidth: 640,
    inputHeight: 640,
    classNames: ['face', 'hand'],
    scoreThreshold: 0.5,
    description:
      'Detects faces + hands, then crops each hand and regresses its 21 keypoints (MediaPipe-style two-stage).',
  },
  {
    id: 'face-detect',
    name: 'Face Detection (YOLO26n · WIDER FACE)',
    shortName: 'Faces',
    type: 'detection',
    modelUrl: '/models/face_yolo26n.tflite',
    inputWidth: 640,
    inputHeight: 640,
    classNames: ['face'],
    scoreThreshold: 0.45,
    description: 'Detects faces and draws bounding boxes. Trained on WIDER FACE.',
  },
  {
    id: 'hand-pose',
    name: 'Hand Keypoints (YOLO26n-pose · 21 pts)',
    shortName: 'Hand Pose',
    type: 'pose',
    modelUrl: '/models/hand_pose_yolo26n.tflite',
    inputWidth: 640,
    inputHeight: 640,
    classNames: ['hand'],
    scoreThreshold: 0.35,
    keypointThreshold: 0.3,
    description: 'Detects hands and draws the 21-point hand skeleton.',
  },
];

export const getModelById = (id: string): ModelDef | undefined =>
  MODELS.find((m) => m.id === id);
