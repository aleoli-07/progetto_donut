import json
import re

import torch
from PIL import Image  # Per caricare l'immagine dal file
from transformers import DonutProcessor, VisionEncoderDecoderModel

# --- IMPOSTAZIONI ---
# 1. Specifica il percorso del tuo modello addestrato
MODEL_PATH = "/home/pc-rs/Scrivania/DatiLinux/donut-finetuned-receipts/best_model"
# 2. Specifica il percorso dell'immagine che vuoi testare
IMAGE_PATH = "/home/pc-rs/Scaricati/img4.jpeg"

# --------------------

# Carica il processore e il modello
print("Caricamento del modello e processore fine-tuned...")
try:
    processor = DonutProcessor.from_pretrained(MODEL_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
except OSError:
    print(f"Errore: Modello non trovato in '{MODEL_PATH}'.")
    exit()

# Sposta il modello sulla GPU se disponibile
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Modello caricato su: {device}")

# Carica l'immagine
print(f"Caricamento immagine da: {IMAGE_PATH}")
try:
    image = Image.open(IMAGE_PATH).convert("RGB")
except FileNotFoundError:
    print(f"Errore: Immagine non trovata in '{IMAGE_PATH}'. Controlla il percorso.")
    exit()

# Prepara l'immagine per il modello
pixel_values = processor(image, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

# Prepara l'input per il decoder (il prompt di inizio)
task_prompt = "<s>"
decoder_input_ids = processor.tokenizer(
    task_prompt, add_special_tokens=False, return_tensors="pt"
).input_ids
decoder_input_ids = decoder_input_ids.to(device)

print("\nEsecuzione dell'inferenza...")
# Genera l'output dal modello
outputs = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=model.config.decoder.max_length,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=4,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

# Decodifica e processa l'output
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
    processor.tokenizer.pad_token, ""
)
sequence = re.sub(
    r"<.*?>", "", sequence, count=1
).strip()  # Rimuove il primo tag <s_...>
prediction = processor.token2json(sequence)

# Mostra i risultati
print("\n--- RISULTATO ESTRAZIONE ---")
print(json.dumps(prediction, indent=2, ensure_ascii=False))
print("\n------------------------------")

# Mostra l'immagine originale
try:
    image.show()
except Exception:
    image.save("single_test_output.png")
    print("Impossibile aprire l'immagine. Salvata come 'single_test_output.png'.")
