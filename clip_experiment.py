import torch, open_clip, numpy as np
from PIL import Image
from roboflow import Roboflow
from utils import *
import json
import os
import torch.nn.functional as F
from rembg import remove
import numpy as np
from prompt_learners import PromptLearnerOpenCLIP, TextEncoderOpenCLIP
 


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


def init_ctx_from_prompts(prompt_learner, clip_model, tokenizer, prompts_per_class, alpha_seed=0.0, seed_text="a photo of a"):
    device = next(clip_model.parameters()).device
    dtype  = next(clip_model.parameters()).dtype
    n_ctx  = prompt_learner.n_ctx

    flat = []
    slices = []
    s = 0
    for plist in prompts_per_class:
        cur = [(p if p.strip().endswith('.') else p.strip()+'.') for p in plist]
        flat.extend(cur)
        e = s + len(cur)
        slices.append((s, e))
        s = e

    with torch.no_grad():
        toks = torch.cat([tokenizer([p]) for p in flat], dim=0).to(device)    # [N,L]
        emb  = clip_model.token_embedding(toks).type(dtype)                    # [N,L,D]
        ctx_chunks = emb[:, 1:1+n_ctx, :]                                      # [N,n_ctx,D]

    per_class_ctx = []
    for s, e in slices:
        per_class_ctx.append(ctx_chunks[s:e].mean(dim=0))                      # [n_ctx,D]

    ctx_init = torch.stack(per_class_ctx, dim=0).mean(dim=0)                   # [n_ctx,D]

    if alpha_seed > 0.0:
        seed_tok = tokenizer([seed_text]).to(device)
        with torch.no_grad():
            seed_emb = clip_model.token_embedding(seed_tok).type(dtype)[0, 1:1+n_ctx, :]
        ctx_init = alpha_seed*seed_emb + (1-alpha_seed)*ctx_init

    with torch.no_grad():
        prompt_learner.ctx.copy_(ctx_init)


def main():
    rf = Roboflow(api_key="B5qYFzGDrNSzMkOluc7f")
    project = rf.workspace("lorenzo-dgkm4").project("indoor-plant-disease-dataset-odg74-ystag")
    dataset = project.version(1).download("folder") 
    path = os.path.abspath("prompts.json")

    with open(path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-B-32"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    text_encoder = TextEncoderOpenCLIP(model).to(device).eval()

    # Data module: dataloader per pianta
    dm = PlantDataModule(
        dataset_path=dataset.location,
        preprocess=preprocess,
        batch_size=16,
        shuffle_train=True,
        num_workers=4,
        pin_memory=True,
        splits=("train", "valid", "test")
    )

    # Congela CLIP
    for p in model.parameters():
        p.requires_grad_(False)

    epochs = 5
    temperature = 0.01  # più tipico per CLIP

    for plant_name in classes.keys():
        # Dataloader e mapping per questa pianta
        train_loader_img = dm.get_train(plant_name)
        maps = dm.get_label_map(plant_name, split="train")
        idx2label = maps["idx2label"]  # es. {0: 'Healthy', 1: 'Bacterial_wilt_disease', ...}

        # Costruisci classnames nell'ordine del dataloader
        class_order = [idx2label[i] for i in range(len(idx2label))]
        classnames = [c.replace("_"," ") for c in class_order]

        # Prompt Learner per questa pianta
        pl = PromptLearnerOpenCLIP(
            classnames=classnames,
            clip_model=model,
            n_ctx=16,
            ctx_init=None,
            class_token_position="end",
            model_name=model_name          # <-- stringa corretta
        ).to(device)

        # Inizializza i context token dai prompt JSON (stesso ordine class_order)
        prompts_per_class = [classes[plant_name]["diseases"][k] for k in class_order]
        init_ctx_from_prompts(pl, model, tokenizer, prompts_per_class, alpha_seed=0.3)

        # Optim solo sul PL
        opt = torch.optim.AdamW(pl.parameters(), lr=1e-3, weight_decay=0.0)

        pl.train()
        for ep in range(1, epochs+1):
            tot, ok, loss_sum = 0, 0, 0.0
            for b_idx, (xb, yb) in enumerate(train_loader_img):
                # Se vuoi limitarti ai primi 5 batch:
                # if b_idx >= 5: break

                xb, yb = xb.to(device), yb.to(device)
                with torch.no_grad():
                    img_feats = F.normalize(model.encode_image(xb), dim=-1)

                # Text feats (senza no_grad: serve il grafo per pl.ctx)
                prompts_embeds, tok_prompts = pl()
                text_feats = text_encoder(prompts_embeds, tok_prompts)
                text_feats = F.normalize(text_feats, dim=-1)

                logits = (img_feats @ text_feats.t()) / temperature
                loss = F.cross_entropy(logits, yb)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                loss_sum += loss.item() * xb.size(0)
                ok += (logits.argmax(1) == yb).sum().item()
                tot += xb.size(0)

            print(f"[{plant_name}] epoch {ep}/{epochs} | loss {loss_sum/tot:.4f} | acc {ok/tot:.3f}")

        # (facoltativo) salva i context token per questa pianta
        # torch.save(pl.state_dict(), f"{plant_name}_prompt_learner.pt")


        """embed_texts = [p for disease_prompts in class_text_embeds[plant_name].values() for p in disease_prompts]
        pred_class, prob = predict_image(img, list(classes[plant_name]["diseases"].keys()), embed_texts, device, model, preprocess)
        print(f"predicted: {pred_class}, actual: {condition}, with prob: {prob}")"""

if __name__ == "__main__":
    main()
