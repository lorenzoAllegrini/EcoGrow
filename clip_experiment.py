import torch, open_clip, numpy as np
from PIL import Image
from roboflow import Roboflow
from utils import *
import json
import os

rf = Roboflow(api_key="B5qYFzGDrNSzMkOluc7f")
print(rf.workspace())
project = rf.workspace("lorenzo-dgkm4").project("indoor-plant-disease-dataset-odg74-ystag")
print(project.versions())
dataset = project.version(1).download("folder") 

path = os.path.abspath("prompts.json")

with open(path, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

texts = get_img_label_pairs(dataset.location)

with torch.no_grad():
    t = tokenizer(texts).to(device)
    tfeat = model.encode_text(t)
    tfeat /= tfeat.norm(dim=-1, keepdim=True)

def predict_zero_shot(img_path, abstain=0.28, topk=3):
    img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        ifeat = model.encode_image(img); ifeat /= ifeat.norm(dim=-1, keepdim=True)
        sims = (ifeat @ tfeat.T).softmax(dim=-1).cpu().numpy()[0]

    # aggrega per classe (media delle varianti)
    scores = {}
    i = 0
    for k, plist in PROMPTS.items():
        n = len(plist)
        scores[k] = float(np.mean(sims[i:i+n])); i += n

    # ordina e applica astensione
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if ranked[0][1] < abstain:
        return {"label":"unknown", "score":ranked[0][1], "topk":ranked[:topk], "scores":scores}
    return {"label":ranked[0][0], "score":ranked[0][1], "topk":ranked[:topk], "scores":scores}
