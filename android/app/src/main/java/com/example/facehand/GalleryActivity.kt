package com.example.facehand

import android.app.AlertDialog
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.GridView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * The face gallery screen: a grid of every enrolled identity (its aligned 112x112 crop + label +
 * how many times it's been seen). Tap the **✕** badge on a face to delete it; **long-press** a face
 * to rename it (so a stray tap never renames by accident); **Clear all** in the header wipes the
 * gallery. UI is built in code — no extra layout/adapter resources.
 */
class GalleryActivity : AppCompatActivity() {

    private lateinit var gallery: FaceGallery
    private lateinit var header: TextView
    private lateinit var adapter: Adapter
    private var items: List<FaceGallery.Person> = emptyList()
    private val deleteBtnId = View.generateViewId() // stable id to find the ✕ on cell reuse

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        gallery = FaceGallery.get(this)

        val density = resources.displayMetrics.density
        val pad = (12 * density).toInt()

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
            numColumns = 3
            horizontalSpacing = (8 * density).toInt()
            verticalSpacing = (8 * density).toInt()
            setPadding(pad, 0, pad, pad)
            isVerticalScrollBarEnabled = true
        }
        adapter = Adapter(density)
        grid.adapter = adapter
        // Tap does nothing (no accidental rename); long-press renames; the ✕ badge deletes.
        grid.setOnItemLongClickListener { _, _, pos, _ -> renameAt(pos); true }

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
        header.text = if (items.isEmpty()) "No faces yet — point the camera at someone."
            else "${items.size} ${if (items.size == 1) "person" else "people"}  ·  long-press to rename"
        adapter.notifyDataSetChanged()
    }

    private fun renameAt(pos: Int) {
        val p = items.getOrNull(pos) ?: return
        val input = EditText(this).apply { setText(p.label); setSelection(p.label.length) }
        AlertDialog.Builder(this)
            .setTitle("Rename")
            .setView(input)
            .setPositiveButton("OK") { _, _ ->
                gallery.rename(p, input.text.toString().trim().ifBlank { p.label })
                refresh()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun deletePerson(p: FaceGallery.Person) {
        AlertDialog.Builder(this)
            .setTitle("Delete ${p.label}?")
            .setMessage("Remove this face from the gallery. The camera can re-add it later.")
            .setPositiveButton("Delete") { _, _ -> gallery.delete(p); refresh() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun confirmClearAll() {
        if (items.isEmpty()) return
        AlertDialog.Builder(this)
            .setTitle("Clear all ${items.size} faces?")
            .setMessage("Removes every enrolled face. The camera can re-add them later.")
            .setPositiveButton("Clear all") { _, _ -> gallery.clear(); refresh() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    /** Grid cell = aligned face thumbnail (with a big ✕ delete badge) above a label. */
    private inner class Adapter(private val density: Float) : BaseAdapter() {
        override fun getCount() = items.size
        override fun getItem(pos: Int) = items[pos]
        override fun getItemId(pos: Int) = items[pos].id.toLong()

        override fun getView(pos: Int, convertView: View?, parent: ViewGroup): View {
            val cell = (convertView as? LinearLayout) ?: buildCell()
            val p = items[pos]
            cell.findViewById<ImageView>(android.R.id.icon).setImageBitmap(BitmapFactory.decodeFile(p.thumbPath))
            cell.findViewById<TextView>(android.R.id.text1).text = "${p.label}  ·  ${p.count}"
            cell.findViewById<TextView>(deleteBtnId).setOnClickListener { deletePerson(p) }
            return cell
        }

        private fun buildCell(): LinearLayout {
            val ctx = this@GalleryActivity
            val sz = (104 * density).toInt()
            val q = (8 * density).toInt()
            val frame = FrameLayout(ctx).apply {
                layoutParams = LinearLayout.LayoutParams(sz, sz)
                addView(ImageView(ctx).apply {
                    id = android.R.id.icon
                    layoutParams = FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT,
                    )
                    scaleType = ImageView.ScaleType.CENTER_CROP
                })
                // Big, easy-to-hit delete badge (bigger glyph + generous padding = larger touch target).
                addView(TextView(ctx).apply {
                    id = deleteBtnId
                    text = "✕"
                    setTextColor(Color.WHITE)
                    textSize = 18f
                    isClickable = true
                    setPadding(q, q / 2, q, q / 2)
                    setBackgroundColor(Color.parseColor("#CC000000"))
                    layoutParams = FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT,
                        Gravity.TOP or Gravity.END,
                    )
                })
            }
            return LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                addView(frame)
                addView(TextView(ctx).apply {
                    id = android.R.id.text1
                    setTextColor(Color.WHITE)
                    textSize = 12f
                    gravity = Gravity.CENTER
                    setPadding(0, (4 * density).toInt(), 0, 0)
                })
            }
        }
    }
}
