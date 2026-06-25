package com.example.facehand

import android.app.AlertDialog
import android.app.Dialog
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.FrameLayout
import android.widget.GridView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File

/**
 * The framing gallery: a 2-column grid of the square snapshots captured when the user makes the
 * two-hand "L" framing gesture. Tap a shot to view it full-screen; tap the ✕ badge (or long-press)
 * to delete. UI is built in code — no extra layout resources.
 */
class FramingGalleryActivity : AppCompatActivity() {

    private lateinit var gallery: FramingGallery
    private lateinit var header: TextView
    private lateinit var adapter: Adapter
    private var items: List<File> = emptyList()
    private val deleteBtnId = View.generateViewId()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        gallery = FramingGallery.get(this)
        val d = resources.displayMetrics.density
        val pad = (12 * d).toInt()

        header = TextView(this).apply {
            setTextColor(Color.WHITE); textSize = 16f; setPadding(pad, pad, pad, pad)
        }
        val clearBtn = TextView(this).apply {
            text = "Clear all"
            setTextColor(Color.parseColor("#ef4444"))
            textSize = 15f; isClickable = true
            setPadding(pad, pad, pad, pad)
            setOnClickListener { confirmClearAll() }
        }
        val headerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(header, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(clearBtn, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        }
        val grid = GridView(this).apply {
            numColumns = 2
            horizontalSpacing = (8 * d).toInt(); verticalSpacing = (8 * d).toInt()
            setPadding(pad, 0, pad, pad)
        }
        adapter = Adapter(d)
        grid.adapter = adapter
        grid.setOnItemClickListener { _, _, pos, _ -> items.getOrNull(pos)?.let(::showFull) }
        grid.setOnItemLongClickListener { _, _, pos, _ -> items.getOrNull(pos)?.let(::confirmDelete); true }

        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#0b0f17"))
            addView(headerRow, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(grid, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        })
    }

    override fun onResume() { super.onResume(); refresh() }

    private fun refresh() {
        items = gallery.all()
        header.text = if (items.isEmpty()) "No shots yet — make the two-hand frame to capture."
            else "Framing · ${items.size} ${if (items.size == 1) "shot" else "shots"}"
        adapter.notifyDataSetChanged()
    }

    /** Full-screen viewer for one shot; tap anywhere to dismiss. */
    private fun showFull(f: File) {
        val bmp = BitmapFactory.decodeFile(f.path) ?: return
        val pad = (16 * resources.displayMetrics.density).toInt()
        val img = ImageView(this).apply {
            setImageBitmap(bmp)
            scaleType = ImageView.ScaleType.FIT_CENTER
        }
        val caption = TextView(this).apply {
            text = gallery.caption(f) ?: "(no description yet)"
            setTextColor(Color.WHITE)
            textSize = 16f
            setPadding(pad, pad, pad, pad)
            setBackgroundColor(Color.parseColor("#CC000000"))
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.BLACK)
            addView(img, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
            addView(caption, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        }
        Dialog(this, android.R.style.Theme_Black_NoTitleBar_Fullscreen).apply {
            setContentView(root)
            root.setOnClickListener { dismiss() }
            setOnDismissListener { img.setImageDrawable(null); bmp.recycle() }
            window?.setBackgroundDrawable(ColorDrawable(Color.BLACK))
            show()
        }
    }

    private fun confirmDelete(f: File) {
        AlertDialog.Builder(this)
            .setTitle("Delete this shot?")
            .setPositiveButton("Delete") { _, _ -> gallery.delete(f); refresh() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun confirmClearAll() {
        if (items.isEmpty()) return
        AlertDialog.Builder(this)
            .setTitle("Clear all ${items.size} shots?")
            .setMessage("Removes every framing shot and its description.")
            .setPositiveButton("Clear all") { _, _ -> gallery.clear(); refresh() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private inner class Adapter(private val d: Float) : BaseAdapter() {
        private val cell = (resources.displayMetrics.widthPixels - (44 * d).toInt()) / 2 // ~half width

        override fun getCount() = items.size
        override fun getItem(pos: Int) = items[pos]
        override fun getItemId(pos: Int) = pos.toLong()

        override fun getView(pos: Int, convertView: View?, parent: ViewGroup): View {
            val frame = (convertView as? FrameLayout) ?: FrameLayout(this@FramingGalleryActivity).apply {
                addView(ImageView(context).apply {
                    id = android.R.id.icon
                    layoutParams = FrameLayout.LayoutParams(cell, cell)
                    scaleType = ImageView.ScaleType.CENTER_CROP
                })
                addView(TextView(context).apply {
                    id = deleteBtnId
                    text = "✕"; setTextColor(Color.WHITE); textSize = 14f; isClickable = true
                    val q = (6 * d).toInt(); setPadding(q, q / 2, q, q / 2)
                    setBackgroundColor(Color.parseColor("#CC000000"))
                    layoutParams = FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT,
                        Gravity.TOP or Gravity.END,
                    )
                })
            }
            val f = items[pos]
            frame.findViewById<ImageView>(android.R.id.icon).setImageBitmap(decodeThumb(f.path, cell))
            frame.findViewById<TextView>(deleteBtnId).setOnClickListener { confirmDelete(f) }
            return frame
        }

        /** Decode a grid thumbnail down-scaled to ~[reqPx] (the shots are megapixel-sized). */
        private fun decodeThumb(path: String, reqPx: Int): Bitmap? {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(path, bounds)
            var sample = 1
            while (bounds.outWidth / (sample * 2) >= reqPx && bounds.outHeight / (sample * 2) >= reqPx) sample *= 2
            return BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = sample })
        }
    }
}
