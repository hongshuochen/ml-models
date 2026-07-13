# Keep TensorFlow Lite classes (it uses native/reflection internally).
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**
