import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ── Model ──────────────────────────────────────────────────────────────────────
model = load_model("my_model.keras")

CLASS_NAMES  = ["Fresh", "Rotten"]
CLASS_COLORS = {"Fresh": "#82C97A", "Rotten": "#D98C6E"}


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_img(pil_img):
    if pil_img is None:
        return "", 0.0, {}, None

    img_resized = pil_img.resize((224, 224))
    img_array   = image.img_to_array(img_resized)
    img_array   = np.expand_dims(img_array, axis=0) / 255.0

    pred      = model.predict(img_array, verbose=0)[0]
    pred_idx  = int(np.argmax(pred))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(pred[pred_idx])

    prob_dict    = {CLASS_NAMES[i]: float(pred[i]) for i in range(len(CLASS_NAMES))}
    
    display_name = f"{'🍃' if pred_label == 'Fresh' else '🍂'}  {pred_label}"

    # ── Chart ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0D0D20")
    ax.set_facecolor("#0D0D20")

    bars = ax.bar(
        CLASS_NAMES,
        [prob_dict[c] for c in CLASS_NAMES],
        color=[CLASS_COLORS[c] for c in CLASS_NAMES],
        width=0.42,
        edgecolor="none",
    )
    for bar, cls in zip(bars, CLASS_NAMES):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.025,
                f"{h * 100:.1f}%", ha="center", va="bottom",
                color="white", fontsize=13, fontweight="bold")

    bars[pred_idx].set_linewidth(2)
    bars[pred_idx].set_edgecolor("white")

    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Probability", color="#6B7A8F", fontsize=11)
    ax.set_title("Model Confidence", color="#E8EAF0", fontsize=14, pad=12,
                 fontweight="bold")
    ax.tick_params(colors="#8892A4", labelsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E1E38")
    plt.tight_layout()

    return display_name, round(confidence * 100, 1),  fig


# ── CSS ────────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

/* ── Global ── */
body, .gradio-container {
    background: #090912 !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
    color: #E8EAF0 !important;
}
.gradio-container {
    width: 100vw !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 24px 40px !important;
}

/* ── App title ── */
#app-title {
    text-align: center;
    margin-bottom: 36px;
}
#app-title h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
    margin-bottom: 8px !important;
}
#app-title p {
    font-size: 1rem !important;
    color: #6B7A8F !important;
}
#app-title .badges {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 12px;
}
#app-title .badge {
    background: #151526;
    border: 1px solid #242438;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #8892A4;
}

/* ── Floating cards ── */
.gr-block, .gr-box, .block {
    background: #111125 !important;
    border: 1px solid #1E1E38 !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 24px 60px rgba(0,0,0,0.55) !important;
}

/* ── Section labels ── */
label span, .label-wrap span {
    color: #5A6478 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Text/number outputs ── */
textarea, input[type="text"], input[type="number"] {
    background: #0D0D20 !important;
    color: #E8EAF0 !important;
    border: 1px solid #1E1E38 !important;
    border-radius: 12px !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    padding: 12px 14px !important;
}

/* ── Predict button ── */
#predict-btn {
    background: #1E4D1E !important;
    border: 1px solid #2E7D2E !important;
    color: #A8D5A2 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
    padding: 15px 0 !important;
    letter-spacing: 0.03em !important;
    transition: transform 0.15s, background 0.15s !important;
}
#predict-btn:hover {
    background: #245724 !important;
    transform: translateY(-2px) !important;
}
#predict-btn:active { transform: scale(0.97) !important; }

.warning {
    color: #ffcc00 !important;
    font-weight: bold;
}

/* ── Upload area ── */
.image-container, .upload-container {
    border: 1.5px dashed #252540 !important;
    background: #0D0D20 !important;
    border-radius: 14px !important;
    transition: border-color 0.2s !important;
}
.image-container:hover { border-color: #82C97A !important; }

/* ── JSON ── */
.json-component {
    background: #0D0D20 !important;
    border: 1px solid #1E1E38 !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    color: #82C97A !important;
}

/* ── Examples ── */
.examples-holder { background: #0D0D20 !important; border-radius: 12px !important; }
.example-btn {
    background: #111125 !important;
    border: 1px solid #1E1E38 !important;
    color: #8892A4 !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    transition: border-color 0.15s !important;
}
.example-btn:hover { border-color: #82C97A !important; color: #82C97A !important; }

/* ── Markdown prose ── */
.prose h3 { color: #F0A96B !important; font-size: 1.05rem !important; font-family: 'Syne', sans-serif !important; }
.prose p, .prose li { color: #5A6A80 !important; font-size: 0.95rem !important; line-height: 1.7 !important; }
.prose strong { color: #A8D5A2 !important; }


footer {
    display: none !important;
}

#root footer {
    display: none !important;
}

/* newer Gradio versions */
.gradio-container footer {
    display: none !important;
}

/* remove “Powered by” strip */
.svelte-1ipelgc {
    display: none !important;
}
"""
# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="FruitSense AI") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="app-title">
        <h1>
            <span style="color:#82C97A">Fruit</span><span style="color:#F0A96B">Sense</span>
            <span style="color:#ffffff"> AI</span>
        </h1>
        <p>Deep-learning powered freshness detection — know your fruit before it's too late</p>
        <div class="badges">
            <span class="badge">CNN · TensorFlow / Keras</span>
            <span class="badge">224 × 224 input</span>
            <span class="badge">Fresh / Rotten</span>
        </div>
    </div>
    """)

    # ── Two-column layout ─────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # LEFT — input
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📤 Upload Fruit Image")
            img_input = gr.Image(type="pil", label="Drag & drop or click to upload", height=260)
            predict_btn = gr.Button("🔍  Analyse Freshness", variant="primary", elem_id="predict-btn")
            gr.Markdown("""
**How it works**
1. Upload any fruit photo
2. CNN resizes to **224 × 224** and scales pixels 0 → 1
3. Softmax produces Fresh / Rotten probabilities
4. Results and chart appear instantly
            """)

        # RIGHT — output
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📊 Prediction Results")
            with gr.Row():
                label_out = gr.Textbox(label="Predicted Class",  interactive=False, placeholder="—", scale=2)
                conf_out  = gr.Number( label="Confidence (%)", interactive=False, precision=1, scale=1)
            plot_out = gr.Plot( label="Confidence Chart")

    # ── Events ────────────────────────────────────────────────────────────────
    outs = [label_out, conf_out,  plot_out]
    predict_btn.click(fn=predict_img, inputs=img_input, outputs=outs)
    img_input.change(  fn=predict_img, inputs=img_input, outputs=outs)

    # ── Examples ──────────────────────────────────────────────────────────────
    gr.Markdown("---\n### 📸 Sample Images — click to try")
    gr.Examples(
        examples=["examples/img1.png", "examples/img2.jpg"],
        inputs=img_input,
        outputs=outs,
        fn=predict_img,
        cache_examples=False,
    )

    # ── Project info ──────────────────────────────────────────────────────────
    gr.Markdown("""
---
### 📌 About This Project

| | |
|---|---|
| 🧠 **Model** | CNN — TensorFlow / Keras |
| 🎯 **Task** | Fruit freshness classification (Fresh vs Rotten) |
| 🍌 **Supported Fruits** | Banana, Mango, Orange, Strawberry |
| ⚙️ **Input** | 224 × 224 RGB, normalised 0 → 1 |
| 📦 **Output** | Class label + softmax confidence score |
| 🚀 **UI** | Gradio Blocks — HuggingFace Spaces ready |

---

### ⚠️ Model Limitation / Scope Warning

- This model is trained **only on 4 fruit categories**
- It will still classify any input image into one of these categories
- For unseen fruits, predictions are based on visual similarity and may not be reliable 
- This is expected behavior for a supervised CNN trained on limited dataset

---

### 🧠 Prediction Behavior

- Model outputs probability distribution over trained classes
- Final prediction = highest probability class
- Confidence score shows model certainty

""")
demo.launch()
