plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.golf"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.golf"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        // QNN (Hexagon NPU) libs are arm64-only; the phone target is arm64. Restrict ABIs so the
        // QNN .so's package cleanly and the APK doesn't carry dead x86/armeabi variants.
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    // Do NOT compress the .tflite model so it can be memory-mapped from the APK.
    androidResources {
        noCompress += "tflite"
    }
    // Extract native libs to nativeLibraryDir as REAL files — QNN's fastrpc file-service must stream
    // libQnnHtpV79Skel.so to the cDSP from disk; with the AGP default (compressed-in-APK) it isn't a
    // file and the HTP skel load fails ("Failed to load skel, error 1002").
    packaging {
        jniLibs {
            useLegacyPackaging = true
            // Keep only the V79 (Snapdragon 8 Elite) HTP skel/stub + the shared HTP/Prepare/System/
            // delegate libs; drop the other Hexagon versions and the unused DSP/GPU QNN backends.
            excludes += listOf(
                "**/libQnnHtpV68Skel.so", "**/libQnnHtpV68Stub.so",
                "**/libQnnHtpV69Skel.so", "**/libQnnHtpV69Stub.so",
                "**/libQnnHtpV73Skel.so", "**/libQnnHtpV73Stub.so",
                "**/libQnnHtpV75Skel.so", "**/libQnnHtpV75Stub.so",
                "**/libQnnHtpV81Skel.so", "**/libQnnHtpV81Stub.so",
                "**/libQnnDsp.so", "**/libQnnDspV66Skel.so", "**/libQnnDspV66Stub.so",
                "**/libQnnGpu.so",
            )
        }
    }
}

dependencies {
    // CameraX — live camera input + analysis.
    val cameraX = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraX")
    implementation("androidx.camera:camera-camera2:$cameraX")
    implementation("androidx.camera:camera-lifecycle:$cameraX")
    implementation("androidx.camera:camera-view:$cameraX")

    // LiteRT — on-device inference (the TFLite successor, `com.google.ai.edge.litert`). Migrated from
    // org.tensorflow:tensorflow-lite:2.17.0: LiteRT reads the SAME .tflite and keeps the TFLite Interpreter
    // API (imports stay org.tensorflow.lite.*), so this is a drop-in. NOTE: the GPU-delegate-via-Interpreter
    // path (litert-gpu) is the V1 line, frozen at 1.4.2 — pin core to 1.4.2 to match. Maxing GPU (ML Drift)
    // and the Qualcomm QNN NPU delegate live on the 2.x CompiledModel API (a later, bigger change).
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    // GPU delegate — offload the detector to the phone GPU (Adreno on Snapdragon). Falls back to CPU.
    implementation("com.google.ai.edge.litert:litert-gpu:1.4.2")
    // Qualcomm QNN — runs the model on the Hexagon NPU (HTP). On the S25 (SM8750) the raw-head model
    // fully delegates (581/581) at ~24 ms @640, vs ~168 ms CPU / ~153 ms GPU (see [[android-golf-npu-deploy]]).
    // `qnn-runtime` bundles the on-device HTP V79 .so's; `qnn-litert-delegate` gives QnnDelegate (a TFLite Delegate).
    implementation("com.qualcomm.qti:qnn-litert-delegate:2.48.0")
    implementation("com.qualcomm.qti:qnn-runtime:2.48.0")

    // AndroidX UI essentials.
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
}
