# EcoGrow

Strumenti e sperimentazioni per l allenamento di prompt CLIP dedicati al riconoscimento di malattie delle piante da appartamento.

## Setup con Poetry
- Assicurati di avere Python 3.10 o 3.11 e Poetry installati.
- Crea l ambiente e installa le dipendenze:

```bash
poetry install
```

- (Opzionale) Esporta variabili d ambiente utili:

```bash
export ECOGROW_DATASET_PATH=/percorso/al/dataset        # default: datasets
export ECOGROW_EMBEDDINGS_DIR=artifacts/embeddings      # abilita il salvataggio degli embedding
export ECOGROW_CLIP_PRETRAINED=laion2b_s34b_b79k        # modello pre addestrato OpenCLIP
```

## Struttura del dataset
Lo script di training si aspetta una cartella con gli split `train/` e `test/`, ciascuno suddiviso per famiglia botanica e malattia, per esempio:

```
datasets/
  train/
    araceae/
      Healthy/
      Leaf_Tip_Necrosis/
  test/
    araceae/
      Healthy/
      Leaf_Tip_Necrosis/
```

Aggiorna `datasets` con il tuo materiale oppure passa un percorso diverso con `--dataset-path`.

## Eseguire l esperimento CLIP con Poetry
1. Verifica il file dei prompt (`experiments/prompts.json` di default) e modifica famiglie, malattie o testi se necessario.
2. Esegui lo script tramite Poetry:

```bash
poetry run python experiments/clip_experiment.py \
  --dataset-path datasets \
  --prompts-config experiments/prompts.json \
  --embeddings-dir artifacts/embeddings \
  --epochs 10 \
  --batch-size 16 \
  --perc-eval 0.2 \
  --lr 5e-3
```

Parametri utili:
- `--dataset-path`: directory radice con gli split train/test.
- `--prompts-config`: JSON con la mappatura famiglia -> {classe: [prompt_iniziale, prompt_testo]}.
- `--embeddings-dir`: se impostato salva gli embedding `.pt` per famiglia.
- `--run-id`: nome della cartella risultati (default: `clip_<nome_file_prompt>`).
- `--temperature`, `--epochs`, `--batch-size`, `--perc-eval`, `--lr`: controllano il fine tuning dei prompt.

I risultati vengono salvati in `experiments/<run-id>/` con un `results.csv` riassuntivo.

## Processo di training
Lo script `experiments/clip_experiment.py` esegue il seguente flusso:
1. **Caricamento configurazioni**: risolve percorso dataset, directory risultati e JSON dei prompt; crea un `run_id` per la sessione.
2. **Inizializzazione del modello**: carica OpenCLIP (`ViT-B-32` di default), congela il backbone e prepara il wrapper per l encoding immagine/testo.
3. **Preparazione dei dati**: per ogni famiglia costruisce `PlantData` con trasformazioni CLIP e segmentazione automatica (`segment_plant_rgba` + crop + compositing su sfondo nero).
4. **Prompt tuning**: genera il contesto iniziale con `compute_init_ctx`, istanzia `PromptLearnerOpenCLIP` e addestra solo i token di prompt tramite `PromptTuningTrainer` usando AdamW (batch size, lr e temperatura configurabili).
5. **Valutazione**: calcola metriche su validation (se `--perc-eval > 0`) e su test; opzionalmente esporta embedding testuali per ogni famiglia.
6. **Output finale**: stampa metriche aggregate, salva `results.csv` e, se richiesto, file `.pt` con embedding e runtime log nella cartella della run.

## Verifica
Dopo il run controlla:
- console log per andamento di loss/accuracy
- `experiments/<run-id>/results.csv` per le metriche aggregate
- la directory `artifacts/embeddings` (se specificata) per gli embedding da usare in inferenza

Per ripetere l esperimento con configurazioni diverse crea un nuovo `--run-id` o rimuovi la cartella corrispondente sotto `experiments/`.
