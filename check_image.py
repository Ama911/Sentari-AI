import os
import re
import torch
import clip
import numpy as np
import cv2
from PIL import Image
from shapely.geometry import box
from shapely.ops import unary_union

# === Settings ===
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"
PATCH_SIZE = 96
STRIDE = 32
THRESHOLD = 0.29
LOG_PATH = "renamed_log.txt"

# Create output folder if needed
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === CLIP Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

prompts = [
    # Clothing prompts
    "black thong", "black lingerie thong", "exposed crotch", "revealing groin",
    "tight underwear", "nsfw underwear avatar", "bikini underwear", "lingerie",
    "sexualized demon avatar", "anime girl in thong", "revealing lower body",
    
    # Symbol prompts
    "swastika", "nazi eagle", "wehrmacht eagle", "kkk badge", "offensive patch"
]

text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# === Rename files to safe format ===
safe_files = []
log = []

for i, filename in enumerate(os.listdir(INPUT_FOLDER)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    original_path = os.path.join(INPUT_FOLDER, filename)

    # Create safe filename
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    if safe_name != filename:
        safe_path = os.path.join(INPUT_FOLDER, safe_name)
        os.rename(original_path, safe_path)
        log.append(f"{filename} → {safe_name}")
        safe_files.append(safe_name)
    else:
        safe_files.append(filename)

# Save rename log
if log:
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("Renamed files:\n" + "\n".join(log))
    print(f"✏️ Renamed {len(log)} files. See {LOG_PATH}")

# === Process Images ===
for filename in sorted(safe_files):
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"flagged_{filename}")

    # Load image safely
    original_cv = cv2.imread(input_path)
    if original_cv is None:
        print(f"⚠️ Skipping unreadable file: {filename}")
        continue

    original_rgb = cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB)
    height, width, _ = original_rgb.shape
    output = original_cv.copy()

    patch_tensors = []
    patch_coords = []
    all_boxes = []
    all_labels = []

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            patch = original_rgb[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            stddev = gray.std()
            brightness = gray.mean()

            if stddev < 8 or brightness > 245 or brightness < 10:
                cv2.rectangle(output, (x, y), (x+PATCH_SIZE, y+PATCH_SIZE), (150, 150, 150), 1)
                continue

            cv2.rectangle(output, (x, y), (x+PATCH_SIZE, y+PATCH_SIZE), (255, 200, 100), 1)
            patch_tensor = preprocess(Image.fromarray(patch)).unsqueeze(0)
            patch_tensors.append(patch_tensor)
            patch_coords.append((x, y))

    print(f"📄 {filename}: {len(patch_tensors)} patches processed")

    flagged = False
    if patch_tensors:
        patch_batch = torch.cat(patch_tensors).to(device)
        with torch.no_grad():
            image_features = model.encode_image(patch_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
            max_scores, best_indices = similarities.max(dim=1)

        for i, (score, sim_vec) in enumerate(zip(max_scores, similarities)):
            x, y = patch_coords[i]
            box_coords = (x, y, x+PATCH_SIZE, y+PATCH_SIZE)
            labels = [prompts[j] for j, val in enumerate(sim_vec) if val.item() > THRESHOLD]

            if labels:
                flagged = True
                all_boxes.append(box(*box_coords))
                all_labels.append(labels)
                print(f"❌ {filename} patch at ({x},{y}) matched: {labels}")

    # === Merge overlapping boxes ===
    merged = []
    if all_boxes:
        unioned = unary_union(all_boxes)
        if unioned.geom_type == 'Polygon':
            unioned = [unioned]
        elif unioned.geom_type == 'MultiPolygon':
            unioned = list(unioned.geoms)

        for group in unioned:
            minx, miny, maxx, maxy = map(int, group.bounds)
            labels_for_group = []
            for lbl_box, labels in zip(all_boxes, all_labels):
                if lbl_box.intersects(group):
                    labels_for_group.extend(labels)
            unique_labels = sorted(set(labels_for_group))
            merged.append(((minx, miny, maxx, maxy), unique_labels))

    for (x1, y1, x2, y2), labels in merged:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for i, label in enumerate(labels):
            y_offset = y1 + 20 + i * 18
            cv2.putText(output, label, (x1 + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(output_path, output)
    print(f"💾 Output saved to: {output_path}\n")
