plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.facehand"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.facehand"
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
    buildFeatures {
        viewBinding = true
    }
    // Do NOT compress the .tflite model so it can be memory-mapped from the APK.
    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    // CameraX — live camera input + analysis.
    val cameraX = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraX")
    implementation("androidx.camera:camera-camera2:$cameraX")
    implementation("androidx.camera:camera-lifecycle:$cameraX")
    implementation("androidx.camera:camera-view:$cameraX")

    // TensorFlow Lite — on-device inference (base Interpreter API). 2.17.0 (not 2.16.1) so the
    // int8 dynamic-range models' hybrid FULLY_CONNECTED op (v12) loads; 2.16.1 only knew up to v11.
    implementation("org.tensorflow:tensorflow-lite:2.17.0")

    // AndroidX UI essentials.
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("com.google.android.material:material:1.12.0")
}
