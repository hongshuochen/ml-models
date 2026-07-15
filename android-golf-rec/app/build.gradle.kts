plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.golfrec"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.golfrec"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        // QNN (Hexagon NPU) libs are arm64-only; the phone target is arm64.
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
    // Extract native libs to nativeLibraryDir (the QNN fastrpc file-service streams the HTP skel to the
    // cDSP from disk); keep only the V79 (Snapdragon 8 Elite) skel/stub + drop the other backends.
    packaging {
        jniLibs {
            useLegacyPackaging = true
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
    // CameraX — live camera + frame analysis + video recording.
    val cameraX = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraX")
    implementation("androidx.camera:camera-camera2:$cameraX")
    implementation("androidx.camera:camera-lifecycle:$cameraX")
    implementation("androidx.camera:camera-view:$cameraX")
    implementation("androidx.camera:camera-video:$cameraX")

    // ML Kit Pose Detection — on-device 33-landmark body pose (the person being filmed).
    implementation("com.google.mlkit:pose-detection:18.0.0-beta5")

    // LiteRT (TFLite successor) — our ball + club_head detector. Runs the raw-head model on the
    // Qualcomm Hexagon NPU via QNN (~18 ms @640 on the S25). Keeps the org.tensorflow.lite Interpreter API.
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    implementation("com.google.ai.edge.litert:litert-gpu:1.4.2")
    // Qualcomm QNN — HTP (Hexagon NPU) delegate + on-device runtime. See [[android-golf-npu-deploy]].
    implementation("com.qualcomm.qti:qnn-litert-delegate:2.48.0")
    implementation("com.qualcomm.qti:qnn-runtime:2.48.0")

    // AndroidX UI essentials.
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
}
