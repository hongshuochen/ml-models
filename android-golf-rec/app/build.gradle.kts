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

    // TensorFlow Lite + GPU delegate — our club_head detector (public third-person model).
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.17.0")

    // AndroidX UI essentials.
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
}
