import gradio as gr
import cv2
import os
import json
import argparse
from ultralytics import YOLO

# =========================
# Args from terminal
# =========================
parser = argparse.ArgumentParser(description="YOLOv8 Auto Labeler")
parser.add_argument("--model", required=True, help="Path to YOLOv8 model (.pt)")
parser.add_argument("--images", required=True, help="Path to image folder")
parser.add_argument("--output", default="outputs", help="Output folder for JSONs")
args = parser.parse_args()

MODEL_PATH = args.model
IMAGE_FOLDER = args.images
OUTPUT_DIR = args.output
IMAGES_PER_PAGE = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

# =========================
# Estado global
# =========================
image_paths = []
current_page = 0

# =========================
# Utils
# =========================
def is_image(f):
    return f.lower().endswith((".png", ".jpg", ".jpeg"))

def annotate_and_save(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    results = model(img)[0]

    shapes = []
    annotated = img.copy()

    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

            shapes.append({
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "shape_type": "rectangle"
            })

    annotation = {
        "imageHeight": h,
        "imageWidth": w,
        "shapes": shapes,
        "flags": {}
    }

    base = os.path.splitext(os.path.basename(img_path))[0]
    with open(os.path.join(OUTPUT_DIR, base + ".json"), "w") as f:
        json.dump(annotation, f, indent=2)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

def label_full_folder():
    if not os.path.isdir(IMAGE_FOLDER):
        return "❌ Invalid image folder"

    paths = sorted([
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if is_image(f)
    ])

    if len(paths) == 0:
        return "⚠️ No images found"

    for p in paths:
        annotate_and_save(p)

    return f"✅ Labeled {len(paths)} images<br>📁 JSON saved in <code>{OUTPUT_DIR}</code>"

# =========================
# 
# =========================
def load_images():
    global image_paths, current_page
    image_paths = sorted([
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if is_image(f)
    ])
    current_page = 0
    return update_view()

def update_view():
    start = current_page * IMAGES_PER_PAGE
    end = start + IMAGES_PER_PAGE

    imgs = [annotate_and_save(p) for p in image_paths[start:end]]

    while len(imgs) < IMAGES_PER_PAGE:
        imgs.append(None)

    return imgs

def next_page():
    global current_page
    if (current_page + 1) * IMAGES_PER_PAGE < len(image_paths):
        current_page += 1
    return update_view()

def prev_page():
    global current_page
    if current_page > 0:
        current_page -= 1
    return update_view()

# =========================
# UI
# =========================
with gr.Blocks(title="YOLOv8 Auto Labeler") as demo:
    gr.Markdown("## 📂 YOLOv8 Auto Labeler")

    load_btn = gr.Button("▶ Load images")

    with gr.Row():
        img1 = gr.Image(height=400, container=True, show_label=False)
        img2 = gr.Image(height=400, container=True, show_label=False)

    with gr.Row():
        img3 = gr.Image(height=400, container=True, show_label=False)
        img4 = gr.Image(height=400, container=True, show_label=False)

    with gr.Row():
        prev_btn = gr.Button("⬅️ Previous")
        next_btn = gr.Button("Next ➡️")
    label_all_btn = gr.Button("🏷️ Label full folder")
    status_box = gr.Markdown()

    load_btn.click(load_images,
                   outputs=[img1, img2, img3, img4])

    next_btn.click(next_page,
                   outputs=[img1, img2, img3, img4])

    prev_btn.click(prev_page,
                   outputs=[img1, img2, img3, img4])
    label_all_btn.click(
        fn=label_full_folder,
        outputs=status_box
    )


demo.launch()
