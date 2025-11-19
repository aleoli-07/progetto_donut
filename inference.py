# inference.py
# --------------------------------------------------------------------------------
# DESCRIZIONE:
# Questo script esegue l'inferenza (o predizione) utilizzando un modello Donut
# precedentemente addestrato. Prende in input il percorso di un'immagine di
# scontrino e restituisce le informazioni estratte in formato JSON.
#
# FUNZIONAMENTO:
# 1. Carica il modello Donut fine-tunato e il relativo processore da una
#    directory specificata.
# 2. Sposta il modello sulla GPU (se disponibile) per un'inferenza più veloce.
# 3. Pre-processa l'immagine di input utilizzando il `DonutProcessor`.
# 4. Esegue il metodo `model.generate()`, che orchestra la codifica dell'immagine
#    e la decodifica sequenziale per produrre la stringa di output.
# 5. Post-processa la stringa generata dal modello:
#    a. Decodifica i token in testo leggibile.
#    b. Rimuove i token speciali (es. `<s>`, `</s>`).
#    c. Utilizza una regular expression per parsare la stringa strutturata
#       (es. `<s_total>12.34</s_total>`) e convertirla in un dizionario Python.
# 6. Stampa il dizionario risultante in formato JSON.
# --------------------------------------------------------------------------------

import json
import re
from pathlib import Path
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

def parse_model_output(model_output_string: str) -> dict:
    """
    Parsa la stringa di output generata dal modello Donut per estrarre
    le informazioni strutturate in un dizionario.

    Args:
        model_output_string (str): La stringa decodificata prodotta dal modello,
                                   che contiene token speciali e coppie chiave-valore.

    Returns:
        dict: Un dizionario Python con le informazioni estratte (es. {"total": "15.00"}).
    """
    # Rimuove i token di inizio (`<s>`) e fine (`</s>`) sequenza e spazi bianchi inutili.
    cleaned_string = model_output_string.replace("<s>", "").replace("</s>", "").strip()

    # Definisce una regular expression per catturare le coppie <s_chiave>valore</s_chiave>.
    # - `r"<s_(\w+)>"`: Trova un tag di apertura come `<s_total>` e cattura "total" (gruppo 1).
    # - `(.*?)`: Cattura qualsiasi carattere (in modo non-greedy) tra i tag (gruppo 2).
    # - `</s_\1>"`: Assicura che il tag di chiusura corrisponda a quello di apertura.
    pattern = r"<s_(\w+)>(.*?)</s_\1>"
    matches = re.findall(pattern, cleaned_string)

    # Crea un dizionario utilizzando le coppie (chiave, valore) trovate.
    # `value.strip()` pulisce eventuali spazi extra intorno al valore estratto.
    extracted_data = {key: value.strip() for key, value in matches}

    return extracted_data


def run_inference(image_path: str, model_path: str, device: str = "cuda") -> dict:
    """
    Funzione principale che esegue l'intero processo di inferenza su una singola immagine.

    Args:
        image_path (str): Percorso del file dell'immagine dello scontrino.
        model_path (str): Percorso della directory contenente il modello e il processore fine-tunati.
        device (str): Device su cui eseguire il modello ("cuda" o "cpu").

    Returns:
        dict: Dizionario con i dati estratti. Restituisce un dizionario vuoto in caso di errore.
    """
    # --- 1. CARICAMENTO MODELLO E PROCESSORE ---
    print(f"Caricamento del modello da: {model_path}...")
    try:
        processor = DonutProcessor.from_pretrained(model_path)
        # `torch_dtype=torch.bfloat16` riduce l'uso di memoria e accelera l'inferenza
        # su hardware compatibile, mantenendo coerenza con il training.
        model = VisionEncoderDecoderModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
    except OSError:
        print(f"ERRORE: Modello non trovato in '{model_path}'.")
        print("Verifica che il percorso sia corretto e che il training sia stato completato.")
        return {}

    # Sposta il modello sul device specificato (GPU se disponibile, altrimenti CPU).
    if device == "cuda" and torch.cuda.is_available():
        model.to(device)
        print("Modello spostato su GPU (cuda).")
    else:
        device = "cpu"
        model.to(device)
        print("CUDA non disponibile. Modello in esecuzione su CPU.")

    # Imposta il modello in modalità valutazione. Questo disattiva layer come Dropout
    # che sono utili solo durante il training.
    model.eval()

    # --- 2. PREPARAZIONE IMMAGINE E INPUT DECODER ---
    print(f"\nCaricamento e preparazione dell'immagine: {image_path}...")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"ERRORE: Immagine non trovata in '{image_path}'.")
        return {}

    # Il processore converte l'immagine in un tensore di pixel (`pixel_values`)
    # normalizzato e ridimensionato come richiesto dal modello.
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

    # Il decoder di Donut richiede un "task prompt" per iniziare la generazione.
    # Per il nostro task, è semplicemente il token di inizio sequenza `<s>`.
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    decoder_input_ids = decoder_input_ids.to(device)

    # --- 3. ESECUZIONE DELL'INFERENZA ---
    print("Esecuzione dell'inferenza (generazione)...")
    # `torch.no_grad()` disabilita il calcolo dei gradienti, riducendo l'uso di memoria
    # e accelerando l'esecuzione, poiché non dobbiamo fare backpropagation.
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=model.config.num_beams, # Usa gli stessi parametri di beam search del training.
            bad_words_ids=[[processor.tokenizer.unk_token_id]], # Evita che il modello generi token sconosciuti.
            return_dict_in_generate=True,
        )

    # --- 4. DECODIFICA E PARSING DEL RISULTATO ---
    print("Decodifica e parsing del risultato...")
    # `batch_decode` converte la sequenza di ID di token generati in una stringa leggibile.
    sequence = processor.batch_decode(outputs.sequences)[0]

    # Rimuove il prompt iniziale per isolare solo l'output effettivo del modello.
    generated_sequence = sequence.replace(processor.tokenizer.bos_token, "").strip()

    # Usa la funzione di parsing per convertire la stringa in un dizionario pulito.
    extracted_data = parse_model_output(generated_sequence)

    return extracted_data


# --- BLOCCO DI ESECUZIONE PRINCIPALE ---
if __name__ == "__main__":
    # Imposta i percorsi per il modello e l'immagine di test.
    MODEL_DIR = "../best_model"
    # IMPORTANTE: Sostituire con il percorso di un'immagine reale.
    IMAGE_TO_TEST = "../progetto_donut/images/000.jpg"

    # Controlli preliminari per aiutare l'utente.
    if not Path(MODEL_DIR).exists():
        print("-" * 50)
        print(f"ATTENZIONE: La cartella del modello '{MODEL_DIR}' non esiste.")
        print("Assicurati di aver eseguito 'train.py' e che il percorso sia corretto.")
        print("-" * 50)
    elif not Path(IMAGE_TO_TEST).exists() or "path/to" in IMAGE_TO_TEST:
        print("-" * 50)
        print(f"ATTENZIONE: Il file immagine '{IMAGE_TO_TEST}' non è valido.")
        print("Modifica la variabile 'IMAGE_TO_TEST' con il percorso di un'immagine reale.")
        print("-" * 50)
    else:
        # Se i percorsi sono validi, esegui l'inferenza.
        result = run_inference(IMAGE_TO_TEST, MODEL_DIR)

        # Stampa il risultato in un formato JSON ben leggibile.
        print("\n--- RISULTATO ESTRAZIONE ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("----------------------------")