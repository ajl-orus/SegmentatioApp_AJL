package com.example.segmentationapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var imageViewOriginal: ImageView
    private lateinit var imageViewSegmented: ImageView
    private lateinit var btnSelectImage: Button
    private lateinit var btnSegment: Button

    private var interpreter: Interpreter? = null
    private var currentBitmap: Bitmap? = null

    private val imgSize = 224
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val inputStream = contentResolver.openInputStream(it)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            currentBitmap = bitmap

            // Mostramos la foto en la imagen original y limpiamos la segmentada
            imageViewOriginal.setImageBitmap(bitmap)
            imageViewSegmented.setImageBitmap(null)

            btnSegment.isEnabled = true
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Conectamos las variables con los IDs del XML
        imageViewOriginal = findViewById(R.id.imageViewOriginal)
        imageViewSegmented = findViewById(R.id.imageViewSegmented)
        btnSelectImage = findViewById(R.id.btnSelectImage)
        btnSegment = findViewById(R.id.btnSegment)

        loadModel()

        btnSelectImage.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        btnSegment.setOnClickListener {
            currentBitmap?.let {
                runInference(it)
            }
        }
    }

    private fun loadModel() {
        try {
            val modelFile = loadModelFile()
            interpreter = Interpreter(modelFile)
        } catch (e: Exception) {
            Toast.makeText(this, "Error al cargar el modelo: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }
    }

    // 3. CORREGIDO: Eliminamos el parámetro 'filename' para quitar la advertencia del IDE
    @Throws(Exception::class)
    private fun loadModelFile(): MappedByteBuffer {
        val filename = "unet_float32.tflite"
        val fileDescriptor = assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun runInference(bitmap: Bitmap) {
        val inputBuffer = preprocessImage(bitmap)

        val outputBuffer = Array(1) { Array(imgSize) { Array(imgSize) { FloatArray(3) } } }

        interpreter?.run(inputBuffer, outputBuffer)

        val resultBitmap = postprocessOutput(outputBuffer, bitmap)

        // Mostramos el resultado SOLO en la imagen segmentada
        imageViewSegmented.setImageBitmap(resultBitmap)
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, imgSize, imgSize, true)
        val buffer = ByteBuffer.allocateDirect(4 * imgSize * imgSize * 3)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(imgSize * imgSize)
        scaledBitmap.getPixels(pixels, 0, imgSize, 0, 0, imgSize, imgSize)

        for (pixel in pixels) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            val rNorm = (r - mean[0]) / std[0]
            val gNorm = (g - mean[1]) / std[1]
            val bNorm = (b - mean[2]) / std[2]

            buffer.putFloat(rNorm)
            buffer.putFloat(gNorm)
            buffer.putFloat(bNorm)
        }
        buffer.rewind()
        return buffer
    }

    private fun postprocessOutput(output: Array<Array<Array<FloatArray>>>, originalBitmap: Bitmap): Bitmap {
        val resultBitmap = Bitmap.createScaledBitmap(originalBitmap, imgSize, imgSize, true)
            .copy(Bitmap.Config.ARGB_8888, true)

        for (y in 0 until imgSize) {
            for (x in 0 until imgSize) {
                val classes = output[0][y][x]

                var maxIdx = 0
                var maxVal = classes[0]
                for (c in 1 until classes.size) {
                    if (classes[c] > maxVal) {
                        maxVal = classes[c]
                        maxIdx = c
                    }
                }

                if (maxIdx != 2) {
                    val overlayColor = when (maxIdx) {
                        0 -> Color.argb(120, 255, 0, 0) // Rojo semitransparente para Mascota
                        1 -> Color.argb(120, 0, 255, 0) // Verde semitransparente para Bordes
                        else -> Color.TRANSPARENT
                    }

                    val originalPixel = resultBitmap.getPixel(x, y)
                    resultBitmap.setPixel(x, y, blendColors(originalPixel, overlayColor))
                }
            }
        }
        return resultBitmap
    }

    private fun blendColors(from: Int, to: Int): Int {
        val alpha = (to shr 24) and 0xFF
        val ratio = alpha / 255.0f
        val invRatio = 1.0f - ratio

        val r = ((to shr 16 and 0xFF) * ratio + (from shr 16 and 0xFF) * invRatio).toInt()
        val g = ((to shr 8 and 0xFF) * ratio + (from shr 8 and 0xFF) * invRatio).toInt()
        val b = ((to and 0xFF) * ratio + (from and 0xFF) * invRatio).toInt()

        return Color.argb(255, r, g, b)
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter?.close()
    }
}