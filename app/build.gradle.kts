plugins {
  alias(libs.plugins.android.application)
    //alias(libs.plugins.kotlin.android) // Solo esta línea para Kotlin
}

android {
    namespace = "com.example.segmentationapp"
    compileSdk = 35 // Sintaxis correcta: un número entero

    // En Android Studio modernos (AGP 8.0+), se usa androidResources en lugar de aaptOptions
    androidResources {
        noCompress += "tflite"
    }

    // NOTA: Si tu Android Studio es muy antiguo y te da error con androidResources,
    // borra el bloque de arriba y usa este en su lugar:
    // aaptOptions {
    //     noCompress += "tflite"
    // }

    defaultConfig {
        applicationId = "com.example.segmentationapp"
        minSdk = 24
        targetSdk = 34 // Bajado a 34 para asegurar compatibilidad con tu SDK instalado
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false // Sintaxis correcta en Kotlin DSL
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    configurations.all {
        resolutionStrategy {
            force("androidx.core:core-ktx:1.12.0")
            force("androidx.activity:activity-ktx:1.8.0")
        }
    }

    // Configuración necesaria para Kotlin
    //kotlinOptions {
    //    jvmTarget = "11"
    //}
}

dependencies {
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)

    // TensorFlow Lite
    implementation("org.tensorflow:tensorflow-lite:2.13.0")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}