import json
import os
from collections import defaultdict

import open_clip
import torch
import torch.nn.functional as F
from roboflow import Roboflow
import torchvision.transforms as T

from image_segmentator import black_bg_composite, crop_to_alpha_bbox, segment_plant_rgba
from prompt_learners import PromptLearnerOpenCLIP, TextEncoderOpenCLIP
from trainers import PromptTuningTrainer
from utils import PlantDataModule, make_segment_fn
 


def encode_texts(prompts, tokenizer, model, device):
    assert isinstance(prompts, list) and all(isinstance(p, str) for p in prompts)
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
    return text_features

def predict_image(
    img,
    class_names,
    all_text_embeds,
    device,
    model,
    preprocess,
    temperature: float = 0.01,
    unknown_threshold: float = 0.5,
    segment_fn=None,
):
    if segment_fn is not None:
        img = segment_fn(img)
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
        prompt_config = json.load(f)

    species_prompts = prompt_config.get("species", {})
    species_to_family = prompt_config.get("species_to_family", {})
    family_definitions = prompt_config.get("families", {})
    if not family_definitions and species_prompts:
        fallback = defaultdict(lambda: {"species": [], "diseases": set()})
        for species_name, species_info in species_prompts.items():
            family_name = species_to_family.get(species_name, species_name)
            fallback[family_name]["species"].append(species_name)
            fallback[family_name]["diseases"].update(species_info.get("diseases", {}).keys())
        family_definitions = {
            fam: {
                "species": info["species"],
                "diseases": sorted(info["diseases"]),
            }
            for fam, info in fallback.items()
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "ViT-B-32"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer(model_name)
    model.float()
    model = model.to(device).eval()
    text_encoder = TextEncoderOpenCLIP(model).float().to(device).eval()

    segment_fn = make_segment_fn(
        segment_plant_rgba,
        crop_to_alpha_bbox,
        black_bg_composite,
        pad=12,
    )

    train_augmentations = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        # ⚠️ NIENTE ToTensor, NIENTE Normalize qui
    ])

    dm = PlantDataModule(
        dataset_path=dataset.location,
        preprocess=preprocess,
        segment_fn=segment_fn,
        batch_size=1,
        shuffle_train=True,
        num_workers=4,
        pin_memory=True,
        splits=("train", "valid", "test"),
        train_transforms=None,
        species_to_family=species_to_family,
        prompts_config_path=path,
    )

    # Congela CLIP
    for p in model.parameters():
        p.requires_grad_(False)

    epochs = 2
    trainer = PromptTuningTrainer(model, text_encoder, device, temperature=0.07)
    
    results = {}

    for family_name, family_info in family_definitions.items():
        family_diseases = family_info.get("diseases", [])
        if not family_diseases:
            continue

        train_loader = dm.get_train(family_name=family_name, shots=10, per_class=False)
        try:
            val_loader = dm.get_val(family_name=family_name)
        except ValueError:
            val_loader = None
        try:
            test_loader = dm.get_test(family_name=family_name)
        except ValueError:
            test_loader = None

        label_maps = dm.get_label_map(None, split="train", family_name=family_name)
        idx2label = label_maps["idx2label"]
        class_order = [idx2label[i] for i in range(len(idx2label))]
        undefined_conditions = [cls for cls in class_order if cls not in family_diseases]
        if undefined_conditions:
            raise KeyError(
                f"Family '{family_name}' is missing definitions for diseases: {undefined_conditions}"
            )
        classnames = [c.replace("_", " ") for c in class_order]

        prompt_learner = PromptLearnerOpenCLIP(
            classnames=classnames,
            clip_model=model,
            n_ctx=16,
            ctx_init=None,
            class_token_position="end",
            model_name=model_name,
        ).to(device)
        for n, p in prompt_learner.named_parameters():
            p.requires_grad = True

        prompts_per_class = []
        mapped_species = [species for species, fam in species_to_family.items() if fam == family_name]
        family_species = list(dict.fromkeys(family_info.get("species", []) + mapped_species))
        for cls in class_order:
            aggregated_prompts = []
            for species_name in family_species:
                species_diseases = species_prompts.get(species_name, {}).get("diseases", {})
                aggregated_prompts.extend(species_diseases.get(cls, []))
            if not aggregated_prompts:
                raise KeyError(
                    f"Missing textual prompts for class '{cls}' in family '{family_name}'."
                )
            prompts_per_class.append(aggregated_prompts)

        init_ctx_from_prompts(prompt_learner, model, tokenizer, prompts_per_class, alpha_seed=0.3)

        optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=1e-4, weight_decay=0.1)
        history = trainer.fit(
            prompt_learner,
            optimizer,
            train_loader,
            epochs=epochs,
            val_loader=val_loader,
            grad_clip=None,
            log_fn=lambda msg, family=family_name: print(f"[{family}] {msg}"),
        )

        plant_results = {"history": history}
        if test_loader is not None:
            plant_results["test"] = trainer.evaluate(prompt_learner, test_loader)

        results[family_name] = plant_results

    for family_name, plant_results in results.items():
        if "test" in plant_results:
            test_metrics = plant_results["test"]
            print(
                f"[{family_name}] test loss {test_metrics.loss:.4f} "
                f"acc {test_metrics.accuracy:.3f}"
            )

    return results

if __name__ == "__main__":
    main()
