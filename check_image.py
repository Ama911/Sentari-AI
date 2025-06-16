import torch
import clip
import numpy as np
import cv2
from PIL import Image
from shapely.geometry import box
from shapely.ops import unary_union

# === 1. Setup CLIP Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# === 2. Load Image ===
image_path = "roblox_skin.png"
original_cv = cv2.imread(image_path)
original_rgb = cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB)
height, width, _ = original_rgb.shape
output = original_cv.copy()

# === 3. Define Patch Scanning Params ===
patch_size = 96
stride = 32

# === 4. Define Suspicious Text Prompts ===
prompts = [
    "nazi symbol", "swastika", "kkk badge", "racist insignia",
    "offensive patch", "white supremacist symbol", "wehrmacht eagle",
    "reichsadler emblem", "nazi eagle badge", "german military patch",
    "silver nazi eagle", "german eagle insignia", "metal eagle badge on black uniform"
]
text_tokens = clip.tokenize(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# === 5. Scan Patches & Filter ===
patch_tensors = []
patch_coords = []
all_boxes = []
all_labels = []

for y in range(0, height - patch_size + 1, stride):
    for x in range(0, width - patch_size + 1, stride):
        patch = original_rgb[y:y+patch_size, x:x+patch_size]
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        stddev = gray.std()
        brightness = gray.mean()

        if stddev < 8 or brightness > 245 or brightness < 10:
            cv2.rectangle(output, (x, y), (x+patch_size, y+patch_size), (150, 150, 150), 1)
            continue

        cv2.rectangle(output, (x, y), (x+patch_size, y+patch_size), (255, 200, 100), 1)

        patch_pil = Image.fromarray(patch)
        patch_tensor = preprocess(patch_pil).unsqueeze(0)
        patch_tensors.append(patch_tensor)
        patch_coords.append((x, y))

print(f"📦 {len(patch_tensors)} patches processed")

# === 6. Run CLIP on Valid Patches ===
flagged = False
box_label_map = []

if patch_tensors:
    patch_batch = torch.cat(patch_tensors).to(device)
    with torch.no_grad():
        image_features = model.encode_image(patch_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarities = image_features @ text_features.T
        max_scores, best_indices = similarities.max(dim=1)

    threshold = 0.29
    for i, (score, sim_vec) in enumerate(zip(max_scores, similarities)):
        x, y = patch_coords[i]
        box_coords = (x, y, x+patch_size, y+patch_size)

        # collect all prompts over threshold
        label_stack = [prompts[j] for j, val in enumerate(sim_vec) if val.item() > threshold]

        if label_stack:
            flagged = True
            all_boxes.append(box(*box_coords))
            all_labels.append(label_stack)
            print(f"❌ Patch at ({x},{y}) matched: {label_stack}")

# === 7. Merge Overlapping Boxes ===
merged = []
if all_boxes:
    unioned = unary_union(all_boxes)
    if unioned.geom_type == 'Polygon':
        unioned = [unioned]
    elif unioned.geom_type == 'MultiPolygon':
        unioned = list(unioned.geoms)

    for group in unioned:
        minx, miny, maxx, maxy = map(int, group.bounds)

        # gather all labels whose boxes intersect this group
        labels_for_group = []
        for lbl_box, labels in zip(all_boxes, all_labels):
            if lbl_box.intersects(group):
                labels_for_group.extend(labels)

        unique_labels = sorted(set(labels_for_group))
        merged.append(((minx, miny, maxx, maxy), unique_labels))

# === 8. Draw Final Merged Boxes and Labels ===
for (x1, y1, x2, y2), labels in merged:
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for i, label in enumerate(labels):
        y_offset = y1 + 20 + i * 18
        cv2.putText(output, label, (x1 + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# === 9. Save Output ===
output_path = "highlighted_output_merged.png"
cv2.imwrite(output_path, output)

if flagged:
    print(f"\n⚠️ Flagged image. Merged output saved to '{output_path}'")
else:
    print("✅ No offensive patches detected.")
