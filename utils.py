import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=8, dtype=torch.float32, device='cuda'):
        super().__init__()
        self.classnames = classnames
        self.device = device

        # tokenizer e embedding del text encoder di CLIP
        self.model = clip_model
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # dimensione embedding testo (di solito 512)
        self.ctx_dim = self.model.text_projection.shape[0] if hasattr(self.model, "text_projection") else 512

        # contesti learnable per classe: [num_class, n_ctx, ctx_dim]
        self.n_ctx = n_ctx
        self.ctx = nn.Parameter(torch.empty(len(classnames), n_ctx, self.ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx, std=0.02)

        # tokenizzazione dei nomi classe (fissi)
        texts = [name.replace("_", " ") for name in classnames]  # es: "Leaf Tip Necrosis"
        self.name_tokens = self.tokenizer(texts)

        # otteniamo gli embedding dei token “veri” dal text encoder (buffer, no grad)
        with torch.no_grad():
            # embedding lookup layer interno
            self.token_embedding = self.model.token_embedding
            self.name_embeds = self.token_embedding(self.name_tokens).to(device)

    def forward(self):
        # Costruisce le sequenze [SOT] + ctx(1..n_ctx) + name_tokens + [EOT]
        # L'encoder di CLIP aggiunge SOT/EOT internamente in open_clip, quindi costruiamo i “middle tokens”
        B = len(self.classnames)
        # context: [C, n_ctx, D]
        ctx = self.ctx  # learnable
        # name_embeds: [C, L, D]
        name = self.name_embeds
        # concatena su dim token
        prompts_embeds = torch.cat([ctx, name], dim=1)  # [C, n_ctx+L, D]
        return prompts_embeds  # embedding pronti per l'encoder testo

def encode_class_prompts(model, prompt_learner, device):
    # ricrea i token per le frasi “dummy” con n_ctx placeholder (es: "x " * n_ctx + class name)
    # ma più semplice: usa token per class name e sostituisci i primi n_ctx embedding.
    tokens = prompt_learner.name_tokens.to(device)             # [C, L]
    # embedding originali dei token
    token_embeds = model.token_embedding(tokens)               # [C, L, D]
    # prepend dei ctx learnable
    C, L, D = token_embeds.shape
    ctx = prompt_learner.ctx                                   # [C, n_ctx, D]
    text_embeds = torch.cat([ctx, token_embeds], dim=1)        # [C, n_ctx+L, D]

    # costruisci position embeddings coerenti (CLIP usa pos_embed per lunghezza fissa)
    # taglio/padding se serve al max_len del tokenizer
    max_len = model.positional_embedding.shape[0]              # lunghezza massima
    if text_embeds.shape[1] > max_len:
        text_embeds = text_embeds[:, :max_len, :]
    else:
        pad = max_len - text_embeds.shape[1]
        pad_zeros = torch.zeros(C, pad, D, device=device, dtype=text_embeds.dtype)
        text_embeds = torch.cat([text_embeds, pad_zeros], dim=1)

    # aggiungi positional embedding e passa nell’encoder testo
    x = text_embeds + model.positional_embedding               # [C, max_len, D]
    x = x.permute(1, 0, 2)  # NLD -> LND per i transformer di open_clip
    x = model.transformer(x)                                   # [L, C, D]
    x = x.permute(1, 0, 2)  # back to [C, L, D]
    x = model.ln_final(x)
    # prendi il token EOT come rappresentazione (open_clip usa indice del token EOT; qui usiamo il pooling sul ultimo non-zero)
    # semplice: usa media sui primi (n_ctx + L originali) non paddati
    # (più fedele sarebbe individuare l’indice EOT dal tokenizer)
    rep = x.mean(dim=1)                                        # [C, D]
    # proietta
    if hasattr(model, "text_projection"):
        rep = rep @ model.text_projection                      # [C, D_text]
    rep = F.normalize(rep, dim=-1)
    return rep  # [num_class, D]

def train_prompt_learning(model, prompt_learner, dataloader, device, epochs=5, lr=5e-3, temperature=0.02):
    model.eval()  # congela CLIP
    for p in model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(prompt_learner.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for images, labels in dataloader:  # images: list of PIL o tensor già preprocess; labels: int
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                img_f = model.encode_image(images)
                img_f = F.normalize(img_f, dim=-1)  # [B,D]

            # forward prompt learner -> text feats [C,D]
            text_f = encode_class_prompts(model, prompt_learner, device)  # [C,D]

            logits = img_f @ text_f.T         # [B,C]
            loss = F.cross_entropy((logits/temperature), labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            preds = logits.argmax(dim=1)
            total += labels.numel()
            correct += (preds == labels).sum().item()
            loss_sum += loss.item()*labels.numel()

        print(f"epoch {ep+1}: loss={loss_sum/total:.4f}, acc={correct/total:.3f}")

def get_img_dataset(dataset_path, split="valid"):
    split_dir = os.path.join(dataset_path, split)
    res = []
    for folder in os.listdir(split_dir):
        parts = folder.split("_")

        if len(parts) > 2:
            plant_name = "_".join(parts[:2])            # es. "Snake Plant"
            condition = "_".join(parts[2:])             # es. "Leaf Withering"
        else:
            plant_name = parts[0]
            condition = parts[1] if len(parts) > 1 else "Unknown"

        folder_path = os.path.join(split_dir, folder)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).convert("RGB")
            res.append((plant_name, condition, img))
    return res
