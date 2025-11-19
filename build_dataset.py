# -*- coding: utf-8 -*-
"""
Script per la preparazione di un dataset per il fine-tuning del modello Donut.

Questo script esegue le seguenti operazioni:
1. Scansiona le directory fornite per trovare coppie di immagini e file JSON di ground truth (GT).
2. Estrae i dati dai JSON e crea un file `metadata.jsonl` necessario per il caricamento.
3. Carica le immagini e i metadati in un oggetto Dataset utilizzando la libreria `datasets`.
4. Inizializza un processore Donut (tokenizer e image processor) basato su un modello pre-addestrato.
5. Aggiunge al tokenizer i token speciali derivati dalle chiavi dei file JSON.
6. Definisce una funzione di trasformazione che:
   - Pre-processa le immagini (ridimensionamento, normalizzazione).
   - Converte il ground truth JSON in una sequenza di token target per il modello.
   - Tokenizza la sequenza target e crea le etichette (labels) per l'addestramento.
7. Applica questa trasformazione all'intero dataset in modo efficiente.
8. Mescola e divide il dataset processato in set di training e validazione.
9. Salva i dataset pronti per l'addestramento e il processore configurato su disco.

Nota: Questa versione dello script processa il dataset originale senza applicare
tecniche di data augmentation.
"""

# ======================================================================================
# 1. IMPORTAZIONE DELLE LIBRERIE NECESSARIE
# ======================================================================================
import json
import re
from pathlib import Path

import numpy as np
from datasets import (
    Array3D,
    Features,
    Sequence,
    Value,
    load_dataset,
)
from tqdm import tqdm  # Libreria per creare barre di avanzamento, utile per monitorare cicli lunghi

# Import specifico per la gestione delle date, con gestione dell'errore se non installata
try:
    from dateutil.parser import parse
except ImportError:
    print("La libreria 'python-dateutil' non è installata. Esegui: pip install python-dateutil")
    exit()

# Import dalle librerie di Hugging Face per il modello Donut
from transformers import AutoTokenizer, DonutImageProcessor, DonutProcessor

# ======================================================================================
# 2. CONFIGURAZIONE DEI PERCORSI E DELLE VARIABILI GLOBALI
# ======================================================================================
print("--- FASE 1: Configurazione dei percorsi ---")

# Definisce il percorso base relativo alla posizione dello script
base_path = Path(__file__).parent
# Definisce il percorso della directory contenente le immagini del dataset
image_dir = Path("../progetto_donut/images")
# Definisce il percorso della directory contenente i file JSON con i dati di ground truth
gt_dir =  Path("../progetto_donut/gdt")

# ======================================================================================
# 3. SCANSIONE E CARICAMENTO DEL DATASET ORIGINALE
# ======================================================================================
print("\n--- FASE 2: Scansione e caricamento del dataset originale ---")

metadata_list = []  # Lista per contenere i metadati di ogni immagine
all_special_tokens = set()  # Insieme per collezionare tutti i token speciali necessari, evitando duplicati

# Token speciali standard per l'inizio e la fine della sequenza target
task_start_token = "<s>"
eos_token = "</s>"

# Itera su tutti i file JSON nella directory di ground truth
for json_path in tqdm(sorted(gt_dir.glob("*.json")), desc="Scansione JSON"):
    # Cerca l'immagine corrispondente con estensione .jpg, .jpeg o .png
    image_path = next((img for ext in [".jpg", ".jpeg", ".png"] if (img := image_dir / f"{json_path.stem}{ext}").exists()), None)
    
    # Se un'immagine corrispondente viene trovata, processa il file
    if image_path:
        with json_path.open("r", encoding="utf-8") as jf:
            gt_data = json.load(jf)
        
        cleaned_gt = gt_data # In questa versione non viene fatta pulizia, ma la variabile è mantenuta
        
        # Estrae le chiavi dal JSON per creare dinamicamente i token speciali del modello Donut
        # Esempio: per una chiave "nome_cliente", crea i token <s_nome_cliente> e </s_nome_cliente>
        for key in cleaned_gt.keys():
            all_special_tokens.add(f"<s_{key}>")
            all_special_tokens.add(f"</s_{key}>")
            
        # Aggiunge una voce alla lista dei metadati con il nome del file e il contenuto GT come stringa JSON
        metadata_list.append({"file_name": image_path.name, "text": json.dumps(cleaned_gt)})

# Scrive i metadati raccolti in un file 'metadata.jsonl'.
# Questo formato è richiesto da `load_dataset` quando si usa la modalità "imagefolder".
metadata_file_path = image_dir / "metadata.jsonl"
with metadata_file_path.open("w", encoding="utf-8") as f:
    for entry in metadata_list:
        f.write(json.dumps(entry) + "\n")

# Carica il dataset usando la funzione `load_dataset` che legge le immagini dalla cartella
# e le associa ai metadati corrispondenti nel file metadata.jsonl.
original_dataset = load_dataset("imagefolder", data_dir=image_dir, split="train")
print(f"Dataset originale caricato con {len(original_dataset)} esempi.")


# ======================================================================================
# 4. INIZIALIZZAZIONE DEL PROCESSORE E DEFINIZIONE DELLE FUNZIONI DI TRASFORMAZIONE
# ======================================================================================
print("\n--- FASE 3: Costruzione del processore e delle funzioni di trasformazione ---")

# Specifica il nome del modello Donut pre-addestrato da Hugging Face
model_name = "naver-clova-ix/donut-base"

# Inizializza il tokenizer e aggiunge i token speciali raccolti dai file JSON.
# Questi token aiuteranno il modello a riconoscere la struttura dei dati.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"additional_special_tokens": sorted(list(all_special_tokens)) + [task_start_token] + [eos_token]})

# Inizializza il processore di immagini, configurando la dimensione target e altre opzioni.
image_processor = DonutImageProcessor.from_pretrained(model_name)
image_processor.size = {"height": 1280, "width": 960}  # Dimensioni a cui le immagini verranno ridimensionate
image_processor.do_align_long_axis = False # Evita di allineare il lato lungo, mantenendo l'orientamento originale

# Combina il tokenizer e il processore di immagini in un unico oggetto `DonutProcessor`
processor = DonutProcessor(image_processor=image_processor, tokenizer=tokenizer)


def json2token(obj, sort_json_key=True):
    """
    Funzione ricorsiva che converte un oggetto JSON (dizionario o lista) in una stringa
    formattata con i token speciali richiesti dal modello Donut.
    Esempio: {"total": "100"} -> "<s_total>100</s_total>"
    """
    if isinstance(obj, dict):
        output = ""
        # Ordina le chiavi per garantire una rappresentazione consistente
        keys = sorted(obj.keys()) if sort_json_key else obj.keys()
        for k in keys:
            output += f"<s_{k}>" + json2token(obj[k]) + f"</s_{k}>"
        return output
    elif isinstance(obj, list):
        # Se l'oggetto è una lista, unisce gli elementi con un token separatore
        return "<sep/>".join([json2token(item, sort_json_key) for item in obj])
    else:
        # Caso base: restituisce il valore come stringa
        return str(obj)


def transform_and_tokenize(sample, max_length=768, ignore_id=-100):
    """
    Funzione principale che processa un singolo campione (immagine + testo) del dataset.
    Converte l'immagine e il testo nel formato richiesto per l'addestramento del modello.
    """
    # 1. Processa l'immagine: la converte in RGB e la trasforma in un tensore di pixel
    image = sample["image"].convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()
    
    # 2. Processa il testo (ground truth):
    #    - Carica la stringa JSON dal campione
    #    - La converte nella sequenza di token desiderata (es. "<s><s_key>value</s_key></s>")
    gt_dict = json.loads(sample["text"])
    target_sequence = task_start_token + json2token(gt_dict) + eos_token
    
    # 3. Tokenizza la sequenza target:
    #    - Converte la stringa in una sequenza di ID numerici (input_ids).
    #    - Applica padding o troncamento per raggiungere la `max_length` definita.
    input_ids = processor.tokenizer(
        target_sequence, add_special_tokens=False, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt"
    ).input_ids.squeeze(0)
    
    # 4. Crea le etichette (labels) per il calcolo della loss:
    #    - Le etichette sono inizialmente una copia degli input_ids.
    #    - Sostituisce l'ID del token di padding con `ignore_id` (-100).
    #      Questo indica alla funzione di loss di ignorare i token di padding durante il calcolo.
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id
    
    # Restituisce un dizionario con i valori processati, convertiti in formati efficienti (numpy)
    return {"pixel_values": pixel_values.numpy().astype(np.float16), "labels": labels.numpy()}

# Definisce esplicitamente la struttura (Features) del dataset processato.
# Questo aiuta a ottimizzare le operazioni di scrittura su disco e di caricamento.
features = Features({
    "pixel_values": Array3D(dtype="float16", shape=(3, 1280, 960)),
    "labels": Sequence(feature=Value(dtype="int64")),
})


# ======================================================================================
# 5. APPLICAZIONE DELLA TRASFORMAZIONE E SALVATAGGIO DEL DATASET
# ======================================================================================
print("\n--- FASE 4: Trasformazione del dataset ORIGINALE ---")

# Applica la funzione `transform_and_tokenize` a ogni elemento del dataset.
# L'operazione `map` è altamente ottimizzata e può essere parallelizzata.
processed_dataset = original_dataset.map(
    transform_and_tokenize,
    remove_columns=["image", "text"],  # Rimuove le colonne originali non più necessarie
    features=features,                 # Applica la struttura definita in precedenza
    num_proc=4,                        # Numero di processi da usare per la parallelizzazione
    writer_batch_size=100,             # Dimensione dei batch per la scrittura su disco
)

print("\n--- FASE 5: Divisione e Salvataggio dei dati processati ---")

# Mescola il dataset processato per garantire che i dati siano distribuiti casualmente.
# `seed=42` assicura che la mescolata sia riproducibile.
shuffled_dataset = processed_dataset.shuffle(seed=42)

# Divide il dataset in un set di training (90%) e uno di test/validazione (10%).
split_dataset = shuffled_dataset.train_test_split(test_size=0.1, seed=42)

print(f"Dataset diviso in {len(split_dataset['train'])} esempi di training e {len(split_dataset['test'])} di validazione.")

# Salva i due split su disco in formato Arrow per un caricamento successivo veloce.
split_dataset["train"].save_to_disk("processed_dataset_train")
split_dataset["test"].save_to_disk("processed_dataset_validation")

# Salva il processore configurato (con il vocabolario aggiornato).
# Questo è fondamentale per poter caricare il modello fine-tuned in futuro.
output_processor_path = "donut_processor_finetuned"
processor.save_pretrained(output_processor_path)
print(f"Processore salvato correttamente in '{output_processor_path}'.")

# Rimuove il file temporaneo metadata.jsonl, non più necessario dopo il caricamento.
metadata_file_path.unlink()

print("\nPreparazione del dataset originale completata con successo!")