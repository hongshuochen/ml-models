/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The TFLite runtime + WASM and the .tflite models are served as static
  // files from /public/vendor and /public/models and loaded at runtime inside
  // a classic Web Worker via importScripts(). Nothing about TensorFlow is
  // bundled by Next, which keeps the app bundle small and avoids bundler/WASM
  // friction.
  //
  // COOP/COEP make the page cross-origin-isolated, which enables SharedArrayBuffer
  // and lets the TFLite WASM runtime use multiple threads (a big speedup for the
  // YOLO models). All app assets are same-origin, so `require-corp` is safe.
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
          { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
        ],
      },
    ];
  },
};

export default nextConfig;
