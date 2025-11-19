# build_dataset_augmented.py (Versione Corretta e Commentata)

# --- IMPORTAZIONE DELLE LIBRERIE NECESSARIE ---
import json
from pathlib import Path
import os

# Librerie per l'elaborazione di immagini e dati numerici
import numpy as np
import cv2  # OpenCV per le trasformazioni di base delle immagini usate da Albumentations
import albumentations as A  # Libreria per l'data augmentation delle immagini
from PIL import Image  # Libreria per la gestione delle immagini

# Librerie del framework Hugging Face
from datasets import (
    Array3D,
    Features,
    Sequence,
    Value,
    load_dataset,
)
from tqdm import tqdm  # Utility per mostrare barre di avanzamento
from transformers import AutoTokenizer, DonutImageProcessor, DonutProcessor

# --- FASE 1 & 2: SCANSIONE DEI FILE E CARICAMENTO DEL DATASET INIZIALE ---
print("\n--- FASE 1&2: Scansione e caricamento del dataset originale ---")

# Definiamo i percorsi di base per le immagini e i file di ground truth (gdt)
base_path = Path(__file__).parent
image_dir = Path("../progetto_donut/images")  # Directory contenente le immagini
gt_dir = Path("../progetto_donut/gdt")      # Directory contenente i file JSON con le annotazioni

# Inizializziamo le strutture dati per contenere i metadati e i token speciali
metadata_list, all_special_tokens = [], set()

# Definiamo i token di inizio e fine sequenza, standard per molti modelli Seq2Seq
task_start_token, eos_token = "<s>", "</s>"

# Iteriamo su tutti i file JSON nella directory di ground truth
for json_path in tqdm(sorted(gt_dir.glob("*.json")), desc="Scansione JSON"):
    # Cerchiamo l'immagine corrispondente al file JSON (con estensioni .jpg, .jpeg, .png)
    image_path = next((img for ext in [".jpg", ".jpeg", ".png"] if (img := image_dir / f"{json_path.stem}{ext}").exists()), None)
    
    if image_path:
        # Se l'immagine esiste, apriamo e carichiamo il file JSON
        with json_path.open("r", encoding="utf-8") as jf:
            gt_data = json.load(jf)
        
        # Estraiamo dinamicamente i token speciali dalle chiavi del JSON.
        # Il modello Donut usa token come <s_chiave> e </s_chiave> per interpretare la struttura del JSON.
        cleaned_gt = gt_data
        for key in cleaned_gt.keys():
            all_special_tokens.add(f"<s_{key}>")
            all_special_tokens.add(f"</s_{key}>")
            
        # Aggiungiamo una voce alla lista dei metadati con il nome del file e il contenuto JSON come stringa
        metadata_list.append({"file_name": image_path.name, "text": json.dumps(cleaned_gt)})

# Creiamo un file "metadata.jsonl" che è richiesto dalla funzione `load_dataset`
# Questo file mappa ogni immagine al suo ground truth testuale.
metadata_file_path = image_dir / "metadata.jsonl"
with metadata_file_path.open("w", encoding="utf-8") as f:
    for entry in metadata_list:
        f.write(json.dumps(entry) + "\n")

# Carichiamo il dataset utilizzando la funzione `imagefolder` di Hugging Face,
# che legge le immagini e le associa al testo contenuto in metadata.jsonl.
original_dataset = load_dataset("imagefolder", data_dir=image_dir, split="train")
print(f"Dataset originale caricato con {len(original_dataset)} esempi.")


# --- FASE 3: COSTRUZIONE DEL PROCESSORE E DELLA PIPELINE DI AUGMENTATION ---
print("\n--- FASE 3: Costruzione del processore e della pipeline di augmentation ---")

# Definiamo il nome del modello Donut pre-addestrato da cui partire
model_name = "naver-clova-ix/donut-base"

# Inizializziamo il tokenizer e aggiungiamo i token speciali che abbiamo raccolto.
# Questo è un passo fondamentale per permettere al modello di comprendere la struttura dei nostri dati.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"additional_special_tokens": sorted(list(all_special_tokens)) + [task_start_token] + [eos_token]})

# Inizializziamo il processore di immagini, configurando la dimensione di input richiesta dal modello.
# `do_align_long_axis = False` evita che l'immagine venga ruotata automaticamente.
image_processor = DonutImageProcessor.from_pretrained(model_name)
image_processor.size = {"height": 1280, "width": 960}
image_processor.do_align_long_axis = False

# Creiamo il `DonutProcessor`, che unisce tokenizer e image processor in un unico oggetto.
processor = DonutProcessor(image_processor=image_processor, tokenizer=tokenizer)


# --- INIZIO PIPELINE DI AUGMENTATION (CORRETTA) ---
# Definiamo una pipeline di trasformazioni di immagine usando Albumentations.
# Queste trasformazioni vengono applicate casualmente solo al set di training
# per aumentare la variabilità dei dati e rendere il modello più robusto.
transform = A.Compose([
    # Applica una leggera rotazione all'immagine.
    A.Rotate(limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    
    # Applica una leggera distorsione prospettica.
    A.Perspective(scale=(0.01, 0.05), p=0.5),
    
    # Simula la compressione JPEG, riducendo la qualità dell'immagine.
    A.ImageCompression(quality_lower=70, quality_upper=95, p=0.5),
    
    # Aggiunge rumore gaussiano all'immagine.
    A.GaussNoise(p=0.3),
    
    # Applica una sfocatura gaussiana.
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    
    # Modifica casualmente luminosità e contrasto.
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
])
# --- FINE PIPELINE DI AUGMENTATION ---


# --- FUNZIONI DI ELABORAZIONE ---

def json2token(obj, sort_json_key=True):
    """
    Funzione ricorsiva che converte un oggetto JSON (dizionario) in una sequenza di token
    in formato stringa, come richiesto dal modello Donut.
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


def apply_transformations(sample, apply_aug=False):
    """
    Funzione che processa un singolo campione (immagine + testo) del dataset.
    Converte l'immagine in tensore e il testo in una sequenza di ID di token.
    Applica l'augmentation se `apply_aug` è True.
    """
    # Carica l'immagine e la converte in formato RGB
    image = sample["image"].convert("RGB")
    
    # Applica la pipeline di augmentation solo se richiesto (per il training set)
    if apply_aug:
        image_np = np.array(image)  # Converte l'immagine in un array NumPy
        augmented = transform(image=image_np) # Applica le trasformazioni
        image = Image.fromarray(augmented['image']) # Riconverte l'array in un'immagine PIL

    # Usa il processore per trasformare l'immagine PIL in un tensore di pixel_values
    pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()
    
    # Carica il JSON dal campo "text", lo converte nel formato token di Donut
    # e aggiunge i token di inizio e fine sequenza.
    gt_dict = json.loads(sample["text"])
    target_sequence = task_start_token + json2token(gt_dict) + eos_token
    
    # Tokenizza la sequenza target, applicando padding e troncamento alla lunghezza massima.
    input_ids = processor.tokenizer(
        target_sequence, add_special_tokens=False, max_length=768, padding="max_length",
        truncation=True, return_tensors="pt"
    ).input_ids.squeeze(0)
    
    # Crea le "labels" per il training. Sono una copia degli input_ids,
    # ma i token di padding vengono sostituiti con -100 per essere ignorati dalla funzione di costo.
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Restituisce un dizionario con i dati processati, convertiti in formati efficienti (numpy)
    return {"pixel_values": pixel_values.numpy().astype(np.float16), "labels": labels.numpy()}

# Definiamo la struttura (Features) del dataset processato.
# Questo aiuta `datasets` a gestire i dati in modo efficiente, specialmente per la scrittura su disco.
features = Features({
    "pixel_values": Array3D(dtype="float16", shape=(3, 1280, 960)),
    "labels": Sequence(feature=Value(dtype="int64")),
})


# --- FASE 4 & 5: ESECUZIONE, DIVISIONE DEL DATASET E SALVATAGGIO ---
print("\n--- FASE 4 & 5: Divisione, Trasformazione e Salvataggio ---")

# Mescoliamo e dividiamo il dataset in training (90%) e validation (10%).
# È cruciale farlo PRIMA dell'augmentation per evitare che campioni aumentati
# finiscano nel set di validazione (data leakage).
split_dataset = original_dataset.shuffle(seed=42).train_test_split(test_size=0.1)

# Applichiamo le trasformazioni al set di training, attivando l'augmentation.
# La funzione `.map()` permette di parallelizzare il processo (`num_proc=4`).
print("Applicazione delle augmentation al set di training...")
train_dataset = split_dataset['train'].map(
    apply_transformations,
    fn_kwargs={"apply_aug": True},  # Applica l'augmentation
    remove_columns=["image", "text"], # Rimuove le colonne originali non più necessarie
    features=features,                # Applica la struttura definita
    num_proc=4,                       # Numero di processi da usare in parallelo
    writer_batch_size=100,            # Ottimizza la scrittura su disco
)

# Applichiamo le trasformazioni al set di validazione, SENZA augmentation.
print("Nessuna augmentation per il set di validazione...")
validation_dataset = split_dataset['test'].map(
    apply_transformations,
    fn_kwargs={"apply_aug": False}, # NON applica l'augmentation
    remove_columns=["image", "text"],
    features=features,
    num_proc=4,
    writer_batch_size=100, 
)

print(f"Dataset diviso in {len(train_dataset)} di training e {len(validation_dataset)} di validazione.")

# Salviamo i dataset processati su disco. Saranno pronti per essere caricati direttamente
# nello script di addestramento.
train_dataset.save_to_disk("processed_dataset_train_aug")
validation_dataset.save_to_disk("processed_dataset_validation_aug")

# Salviamo anche il processore (che ora include i token speciali aggiunti).
# In questo modo, durante l'addestramento e l'inferenza, avremo la certezza di usare
# lo stesso identico vocabolario e le stesse impostazioni di pre-processing.
output_processor_path = "donut_processor_finetuned_aug"
processor.save_pretrained(output_processor_path)
print(f"Processore salvato correttamente in '{output_processor_path}'.")

# Rimuoviamo il file di metadati temporaneo che non è più necessario.
if metadata_file_path.exists():
    metadata_file_path.unlink()
    
print("\nPreparazione del dataset con augmentation completata!")