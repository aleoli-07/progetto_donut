# check_dataset.py
# --------------------------------------------------------------------------------
# DESCRIZIONE:
# Questo script è uno strumento di diagnostica per verificare l'integrità e la
# correttezza di un dataset processato per il modello Donut. Esegue una serie
# di controlli su un singolo campione del dataset per assicurarsi che i dati
# siano nel formato atteso dal modello prima di avviare il training.
#
# FUNZIONAMENTO:
# 1. Carica il dataset di training e il processore salvati su disco.
# 2. Estrae il primo campione (`train_dataset[0]`) per l'ispezione.
# 3. Controlla i `pixel_values` (dati dell'immagine):
#    - Verifica che il tipo di dati sia `float16` (se il pre-processing è stato
#      fatto con questa precisione).
#    - Verifica che la forma (shape) del tensore sia corretta, ad esempio
#      (3, 1280, 960) per un'immagine a 3 canali di 1280x960 pixel.
# 4. Controlla le `labels` (dati di testo tokenizzati):
#    - Verifica che il tipo di dati sia `int64`, come richiesto da PyTorch per
#      gli indici del vocabolario.
#    - Stampa la lunghezza della sequenza di token.
# 5. Esegue un controllo di sanità "umano":
#    - Decodifica la sequenza di `labels` per trasformarla nuovamente in testo.
#    - Sostituisce il valore `-100` (usato dal trainer per ignorare i token
#      durante il calcolo della loss) con il `pad_token_id` per una corretta
#      decodifica.
#    - Controlla che il testo decodificato inizi e finisca con i token speciali
#      `<s>` e `</s>`, confermando che la sequenza è stata costruita correttamente.
# --------------------------------------------------------------------------------

from datasets import load_from_disk
from transformers import DonutProcessor
import numpy as np # Necessario per controllare i tipi e le forme degli array.

print("--- Inizio Verifica Dataset Processato ---")

# --- 1. CARICAMENTO DATASET E PROCESSORE ---
try:
    # Carica il dataset di training salvato localmente.
    train_dataset = load_from_disk("./processed_dataset_train")
    # Carica il processore corrispondente.
    processor = DonutProcessor.from_pretrained("./donut_processor_finetuned")
    print("Dataset e processore caricati con successo.")
except FileNotFoundError as e:
    print(f"ERRORE: File non trovato. Assicurati di aver eseguito lo script di preparazione del dataset. Dettagli: {e}")
    exit()

# --- 2. ESTRAZIONE DI UN CAMPIONE ---
# Esaminiamo il primo elemento come rappresentativo dell'intero dataset.
sample = train_dataset[0]
print(f"\nIspezione del primo campione (indice 0)...")

# --- 3. CONTROLLO DEI DATI DELL'IMMAGINE (`pixel_values`) ---
# Converte la lista di liste in un array NumPy per facilitare l'ispezione.
pixel_values = np.array(sample['pixel_values'])
print(f"\n--- Controllo 'pixel_values' ---")
print(f"  - Tipo di dati: {pixel_values.dtype}")
print(f"  - Forma (Shape): {pixel_values.shape} -> (Canali, Altezza, Larghezza)")

# VERIFICA ATTESA: La forma deve corrispondere alla dimensione di input del modello
# e il tipo di dato a quello usato nel pre-processing (es. float16 per l'efficienza).
if pixel_values.shape == (3, 1280, 960) and pixel_values.dtype == np.float16:
    print("  - VERIFICA SHAPE E TIPO: OK!")
else:
    print(f"  - ATTENZIONE: Formato non corretto! Atteso Shape (3, 1280, 960) e Tipo float16.")
    print(f"    Ottenuto: Shape {pixel_values.shape}, Tipo {pixel_values.dtype}")

# --- 4. CONTROLLO DELLE ETICHETTE (`labels`) ---
# Converte la lista di etichette in un array NumPy.
labels = np.array(sample['labels'])
print(f"\n--- Controllo 'labels' ---")
print(f"  - Tipo di dati: {labels.dtype}")
print(f"  - Lunghezza sequenza: {len(labels)}")

# VERIFICA ATTESA: Le etichette devono essere di tipo `int64` (o `long` in PyTorch).
if labels.dtype == np.int64:
    print("  - VERIFICA TIPO DATI: OK!")
else:
    print(f"  - ATTENZIONE: Tipo di dato non corretto! Atteso 'int64', ottenuto {labels.dtype}")

# --- 5. CONTROLLO DI SANITÀ TRAMITE DECODIFICA ---
print("\n--- Controllo Decodifica Etichette ---")
# Per decodificare correttamente, dobbiamo sostituire l'indice di "ignore" (-100)
# con l'ID del token di padding, altrimenti il decodificatore solleverebbe un errore.
labels_for_decoding = labels.copy()
labels_for_decoding[labels_for_decoding == -100] = processor.tokenizer.pad_token_id
# Usa il metodo `decode` del processore per convertire gli ID in stringa.
decoded_text = processor.decode(labels_for_decoding)

print("  - Testo decodificato dalle etichette:")
print(f"    '{decoded_text}'")

# VERIFICA ATTESA: La sequenza di output per Donut deve essere formattata correttamente
# con i token di inizio e fine sequenza.
if decoded_text.strip().startswith("<s>") and "</s>" in decoded_text:
     print("  - VERIFICA FORMATO: OK! La sequenza contiene i token speciali corretti.")
else:
     print("  - ATTENZIONE: La sequenza decodificata non ha il formato atteso (<s>...</s>).")

print("\n--- Verifica Completata ---")