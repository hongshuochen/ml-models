import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Webcam · YOLO — Real-time on-device inference',
  description:
    'Run your YOLO26 face-detection and hand-pose TFLite models on the webcam feed, fully on-device.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
