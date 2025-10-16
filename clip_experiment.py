import torch, open_clip, numpy as np
from PIL import Image
from roboflow import Roboflow
from utils import *
import json
import os
import torch.nn.functional as F

# utils_seg.py
from rembg import remove
from PIL import Image
import numpy as np

def segment_plant_rgba(pil_img: Image.Image) -> Image.Image:
    """
    Usa U²-Net via rembg per ottenere RGBA con alpha della pianta segmentata.
    """
    pil_img = pil_img.convert("RGB")
    out = remove(pil_img)  # RGBA con alpha (pianta=opaca, bg=trasparente)
    return out

def crop_to_alpha_bbox(rgba_img: Image.Image, pad: int = 8) -> Image.Image:
    """
    Croppa tight sulla bounding box dei pixel non trasparenti.
    """
    arr = np.array(rgba_img)  # H,W,4
    alpha = arr[..., 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return rgba_img.convert("RGB")  # fallback
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(arr.shape[1]-1, x2 + pad); y2 = min(arr.shape[0]-1, y2 + pad)
    cropped = rgba_img.crop((x1, y1, x2+1, y2+1))
    return cropped.convert("RGB")

def black_bg_composite(rgba_img: Image.Image) -> Image.Image:
    """
    Converte RGBA in RGB ponendo lo sfondo a nero (spesso aiuta CLIP).
    """
    bg = Image.new("RGB", rgba_img.size, (0, 0, 0))
    bg.paste(rgba_img, mask=rgba_img.split()[-1])  # alpha
    return bg


def encode_texts(prompts, tokenizer, model, device):
    assert isinstance(prompts, list) and all(isinstance(p, str) for p in prompts)
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
    return text_features

def predict_image(img, class_names, all_text_embeds, device, model, preprocess, temperature=0.01, unknown_threshold=0.5, use_seg=True):
    if use_seg:
        rgba = segment_plant_rgba(img)
        rgba = crop_to_alpha_bbox(rgba, pad=12)
        img = black_bg_composite(rgba)  # oppure lascia rgba e converti in RGB dopo il crop
    else:
        img = img.convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)
    all_text_embeds = torch.stack(all_text_embeds, dim=0) 
    all_text_embeds = F.normalize(all_text_embeds, dim=-1)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        image_features = model.encode_image(img)
        image_features = F.normalize(image_features, dim=-1)
        logits = image_features @ all_text_embeds.t()
        probs = (logits / temperature).softmax(dim=-1).squeeze(0)

    max_prob, idx = probs.max(dim=0)
    pred_class = class_names[idx.item()]
    if max_prob.item() < unknown_threshold:
        pred_class = "unknown"
    return pred_class, max_prob.item()


def main():
    rf = Roboflow(api_key="B5qYFzGDrNSzMkOluc7f")
    project = rf.workspace("lorenzo-dgkm4").project("indoor-plant-disease-dataset-odg74-ystag")
    dataset = project.version(1).download("folder") 
    path = os.path.abspath("prompts.json")

    with open(path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()


    # Calcolo embedding per ogni classe
    class_text_embeds = {c:{} for c in classes.keys()}
    for idx, cls in enumerate(classes):
        for disease, prompts in classes[cls]["diseases"].items():     
            emb = encode_texts(prompts, tokenizer, model, device)
            class_text_embeds[cls][disease] = emb.mean(0, keepdim=True)

    for plant_name, condition, img in get_img_dataset(dataset.location):
        embed_texts = [p for disease_prompts in class_text_embeds[plant_name].values() for p in disease_prompts]
        pred_class, prob = predict_image(img, list(classes[plant_name]["diseases"].keys()), embed_texts, device, model, preprocess)
        print(f"predicted: {pred_class}, actual: {condition}, with prob: {prob}")

if __name__ == "__main__":
    main()
