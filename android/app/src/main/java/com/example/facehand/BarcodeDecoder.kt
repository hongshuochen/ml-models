package com.example.facehand

import android.graphics.Bitmap
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.common.InputImage

/**
 * On-device QR / 1-D barcode reader (Google ML Kit, bundled model — fully offline).
 *
 * Our YOLO detector is what *localizes* a code (drawing the qr/barcode box); this class *decodes*
 * the content from a crop of that box. ML Kit scanning is async — [decode] returns immediately and
 * delivers the first non-blank raw value via [onResult] on the main thread (null = nothing read).
 * The caller owns [bmp] and must keep it alive until [onResult] fires (then recycle it there).
 */
class BarcodeDecoder {

    private val scanner = BarcodeScanning.getClient()

    fun decode(bmp: Bitmap, onResult: (String?) -> Unit) {
        val image = InputImage.fromBitmap(bmp, 0)
        scanner.process(image)
            .addOnSuccessListener { codes ->
                onResult(codes.firstOrNull { !it.rawValue.isNullOrBlank() }?.rawValue)
            }
            .addOnFailureListener { onResult(null) }
    }

    fun close() = scanner.close()
}
