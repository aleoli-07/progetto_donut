# inference_checkpoints.py
# --------------------------------------------------------------------------------
# DESCRIZIONE:
# Questa è una versione migliorata dello script `inference.py`. La differenza
# principale è l'introduzione della gestione degli argomenti da riga di comando
# tramite il modulo `argparse`. Questo rende lo script molto più flessibile e
# riutilizzabile, in quanto consente di specificare il modello, l'immagine e
# il device da utilizzare senza dover modificare il codice sorgente.
#
# FUNZIONAMENTO:
# Le funzioni `parse_model_output` e `run_inference` sono identiche alla versione
# base. La novità risiede nel blocco `if __name__ == "__main__":`:
#
# 1. Viene creato un `ArgumentParser` per definire gli argomenti che lo script
#    può accettare dalla riga di comando.
# 2. Vengono definiti tre argomenti:
#    - `--model_path`: (Obbligatorio) Percorso della cartella del modello.
#    - `--image_path`: (Obbligatorio) Percorso dell'immagine da analizzare.
#    - `--device`: (Opzionale) Device da usare ("cuda" o "cpu"), con "cuda"
#      come default.
# 3. Lo script viene eseguito da terminale, passando i valori per questi
#    argomenti. Esempio:
#    `python inference_checkpoints.py --model_path ./path/to/model --image_path ./img.jpg`
# 4. `parser.parse_args()` legge gli argomenti forniti e li rende disponibili.
# 5. La funzione `run_inference` viene chiamata con i valori passati dall'utente.
# --------------------------------------------------------------------------------

import argparse # Modulo per il parsing degli argomenti da riga di comando.
import json
import re
from pathlib import Path
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# La funzione parse_model_output rimane identica.
def parse_model_output(model_output_string: str) -> dict:
    cleaned_string = model_output_string.replace("<s>", "").replace("</s>", "").strip()
    pattern = r"<s_(\w+)>(.*?)</s_\1>"
    matches = re.findall(pattern, cleaned_string)
    extracted_data = {key: value.strip() for key, value in matches}
    return extracted_data

# La funzione run_inference rimane sostanzialmente identica.
def run_inference(image_path: str, model_path: str, device: str = "cuda") -> dict:
    print(f"Caricamento del modello da: {model_path}...")
    try:
        processor = DonutProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
    except OSError:
        print(f"ERRORE: Modello non trovato in '{model_path}'.")
        return {}

    if device == "cuda" and torch.cuda.is_available():
        model.to(device)
    else:
        device = "cpu"
        model.to(device)
    
    print(f"Modello caricato su: {device.upper()}")
    model.eval()

    print(f"\nCaricamento e preparazione dell'immagine: {image_path}...")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"ERRORE: Immagine non trovata in '{image_path}'.")
        return {}

    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    decoder_input_ids = decoder_input_ids.to(device)

    print("Esecuzione dell'inferenza...")
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=model.config.num_beams,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    print("Decodifica e parsing del risultato...")
    sequence = processor.batch_decode(outputs.sequences)[0]
    generated_sequence = sequence.replace(processor.tokenizer.bos_token, "").strip()
    extracted_data = parse_model_output(generated_sequence)

    return extracted_data


# --- BLOCCO DI ESECUZIONE CON GESTIONE DEGLI ARGOMENTI ---
if __name__ == "__main__":
    # 1. Inizializza il parser di argomenti.
    parser = argparse.ArgumentParser(description="Esegue l'inferenza con un modello Donut fine-tunato.")
    
    # 2. Definisce gli argomenti accettati dallo script.
    parser.add_argument(
        "--model_path",
        type=str,
        required=True, # Questo argomento è obbligatorio.
        help="Percorso alla cartella del modello (es. './donut-finetuned-receipts/best_model')."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True, # Questo argomento è obbligatorio.
        help="Percorso all'immagine dello scontrino da testare."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda", # Valore di default se non specificato.
        choices=["cuda", "cpu"], # Limita i valori accettabili.
        help="Device su cui eseguire l'inferenza (default: cuda)."
    )
    
    # 3. Legge gli argomenti forniti dall'utente dalla riga di comando.
    args = parser.parse_args()

    # 4. Esegue la funzione di inferenza passando gli argomenti letti.
    result = run_inference(args.image_path, args.model_path, args.device)

    # 5. Stampa il risultato se l'inferenza ha avuto successo.
    if result:
        print("\n--- RISULTATO ESTRAZIONE ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("----------------------------")