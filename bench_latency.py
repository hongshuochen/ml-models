#!/usr/bin/env python3
"""Uniform TFLite CPU latency benchmark: warmup + timed runs, fixed threads.
Usage: bench_latency.py <model.tflite> [threads] [runs]"""
import sys, time
import numpy as np
import tensorflow as tf

path = sys.argv[1]
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 50

it = tf.lite.Interpreter(model_path=path, num_threads=threads)
it.allocate_tensors()
ind = it.get_input_details()[0]
shape = ind["shape"]
if ind["dtype"] == np.float32:
    x = np.random.rand(*shape).astype(np.float32)
else:
    x = (np.random.rand(*shape) * 255).astype(ind["dtype"])
for _ in range(8):  # warmup
    it.set_tensor(ind["index"], x); it.invoke()
t = time.perf_counter()
for _ in range(runs):
    it.set_tensor(ind["index"], x); it.invoke()
ms = (time.perf_counter() - t) / runs * 1000
print(f"{path.split('/')[-1]:32s} {ms:7.1f} ms  ({threads} threads, {runs} runs)")
