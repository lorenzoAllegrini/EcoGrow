# EcoGrow CLIP Inference – Docker Workflow

## 1. Pre-generare gli embedding

1. Assicurati di avere il dataset locale o le variabili `ROBOFLOW_*` configurate.
2. Lancia il training/prompt tuning esportando gli embedding finiti:
   ```bash
   export ECOGROW_EMBEDDINGS_DIR=$(pwd)/ecoGrow/artifacts
   python ecoGrow/clip_experiment.py
   ```
   Questo scrive un file `.pt` per ogni famiglia (`artifacts/<Family>.pt`) e l'indice `index.json`, pronti per essere copiati nell'immagine.

> Nota: Se hai già dei `PromptLearner` allenati, puoi semplicemente distribuire i file `.pt` corrispondenti nella cartella `ecoGrow/artifacts`.

## 2. Build dell'immagine

Esegui dal progetto:
```bash
docker build -t ecogrow-clip -f ecoGrow/Dockerfile ecoGrow
```

## 3. Avvio del servizio

```bash
docker run -p 8080:8080 ecogrow-clip
```

Oppure, se vuoi montare embedding aggiornati senza rebuild:
```bash
docker run -p 8080:8080 \
  -v "$(pwd)/ecoGrow/artifacts:/app/artifacts:ro" \
  ecogrow-clip
```

## 4. Chiamata dall'app Flutter

- Endpoint: `POST http://<host>:8080/predict`
- Body: `multipart/form-data` con `image` (file) e opzioni query (`family`, `unknown_threshold`).
- Risposta: JSON con `top_prediction`, lista completa e metadati del modello.

Esempio con `curl`:
```bash
curl -X POST "http://localhost:8080/predict?family=Asparagaceae" \
  -F "image=@/percorso/foto.jpg"
```

L'endpoint `/health` restituisce uno stato rapido (`{"status": "ok"}`) utile per i readiness probe.
