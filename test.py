import json
import random
from pathlib import Path

import torch
from datasets import load_from_disk
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# --------------------------------------------------------------------------
# 1. Caricamento del Modello e del Processore
# --------------------------------------------------------------------------
print("Caricamento del modello e processore fine-tuned...")
MODEL_PATH = "/home/pc-rs/Scrivania/DatiLinux/donut-finetuned-receipts/best_model"
try:
    processor = DonutProcessor.from_pretrained(MODEL_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
except OSError:
    print(
        f"Errore: Modello non trovato in '{MODEL_PATH}'. Assicurati che il training sia completato."
    )
    exit()

# Applica configurazioni di generazione per risultati migliori e più stabili
print("Applicazione configurazione di generazione esplicita...")
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Modello caricato su: {device}")

# --------------------------------------------------------------------------
# 2. Caricamento Dati e Selezione Campione
# --------------------------------------------------------------------------
print("Caricamento dei dati processati...")
try:
    # Carichiamo uno dei due set. Il training set ha più campioni per un test casuale più vario.
    processed_dataset = load_from_disk("./processed_dataset_train")
except Exception as e:
    print(f"Errore caricamento dati: {e}")
    print("Assicurati di aver eseguito prima 'prepare_dataset_final.py'.")
    exit()

# Seleziona un campione casuale dal dataset processato
random_idx = random.randint(0, len(processed_dataset) - 1)
test_sample_processed = processed_dataset[random_idx]
print(f"Campione di test selezionato (indice casuale: {random_idx})")

# Carica l'immagine originale corrispondente in modo affidabile usando il file_name
try:
    original_image_filename = Path(test_sample_processed["file_name"]).name
    original_image_path = Path("./images") / original_image_filename
    print(f"Caricamento immagine originale da: {original_image_path}")
    original_image = Image.open(original_image_path).convert("RGB")
except FileNotFoundError:
    print(f"ERRORE: Immagine originale '{original_image_path}' non trovata!")
    exit()
except KeyError:
    print("ERRORE: La colonna 'file_name' non è stata trovata nel dataset processato.")
    print(
        "Assicurati di aver eseguito l'ultima versione di 'prepare_dataset_final.py'."
    )
    exit()


# --------------------------------------------------------------------------
# 3. Funzione di Predizione
# --------------------------------------------------------------------------
def run_prediction(sample, model, processor):
    """Esegue l'inferenza su un singolo campione processato."""
    pixel_values = torch.tensor(sample["pixel_values"]).unsqueeze(0).to(device)
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    print("\nEsecuzione dell'inferenza con prompt di base '<s>'...")

    # model.generate usa automaticamente i parametri del `model.config`
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.config.decoder.max_length,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Decodifica e pulisce la sequenza di output
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    if sequence.startswith(task_prompt):
        sequence = sequence[len(task_prompt) :]

    prediction = processor.token2json(sequence)

    # Prepara anche il target (ground truth) per il confronto
    target_sequence = sample["target_sequence"]
    target_sequence = target_sequence.replace(
        processor.tokenizer.eos_token, ""
    ).replace(processor.tokenizer.pad_token, "")
    if target_sequence.startswith(task_prompt):
        target_sequence = target_sequence[len(task_prompt) :]
    target = processor.token2json(target_sequence)

    return prediction, target


# --------------------------------------------------------------------------
# 4. Esecuzione e Stampa Risultati
# --------------------------------------------------------------------------
prediction, target = run_prediction(test_sample_processed, model, processor)

print("\n--- RISULTATO ---")
print(f"File immagine: {original_image_filename}")

print("\n[RIFERIMENTO (Ground Truth)]")
print(json.dumps(target, indent=2, ensure_ascii=False))

print("\n[PREDIZIONE del Modello]")
print(json.dumps(prediction, indent=2, ensure_ascii=False))
print("\n-----------------")

# Mostra l'immagine originale
print(f"Visualizzazione dell'immagine di test: {original_image_filename}")
try:
    original_image.show()
    print("L'immagine è stata aperta in una nuova finestra.")
except Exception:
    output_filename = f"test_output_{original_image_filename.replace('.jpg', '.png')}"
    original_image.save(output_filename)
    print(f"Impossibile aprire l'immagine. Salvata come '{output_filename}'.")
