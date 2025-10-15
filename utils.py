import os
from PIL import Image
def get_img_label_pairs(dataset_path, split="valid"):
    split_dir = os.path.join(dataset_path, split)
    pairs = []
    print(os.listdir(split_dir))
    res = []
    for folder in os.listdir(split_dir):
        parts = folder.split("_")

        if len(parts) > 2:
            plant_name = " ".join(parts[:2])            # es. "Snake Plant"
            condition = " ".join(parts[2:])             # es. "Leaf Withering"
        else:
            plant_name = parts[0]
            condition = parts[1] if len(parts) > 1 else "Unknown"
        
        res.extend([(plant_name,condition, Image.open(img_file).convert("RGB")) for img_file in os.listdir(os.path.join(split_dir, folder))])
        
                
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if os.path.exists(label_path):
            with open(label_path) as f:
                label = f.read().strip()
            img_path = os.path.join(img_dir, img_file)
            pairs.append((img_path, label))
    return pairs