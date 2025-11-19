# check_augmentations.py
# --------------------------------------------------------------------------------
# DESCRIZIONE:
# Questo script è uno strumento di visualizzazione e debug per la pipeline di
# data augmentation definita con la libreria `albumentations`. Il suo scopo è
# permettere di verificare visivamente l'effetto delle trasformazioni applicate
# alle immagini prima di avviare un lungo processo di training.
#
# FUNZIONAMENTO:
# 1. Definisce la stessa pipeline di augmentation usata nello script di
#    preparazione del dataset.
# 2. Carica il dataset di immagini originale (non aumentato).
# 3. Crea una directory di output per salvare gli esempi.
# 4. Estrae un numero definito (`NUM_SAMPLES`) di immagini casuali dal dataset.
# 5. Per ogni immagine:
#    a. Salva una copia dell'immagine originale.
#    b. Applica la pipeline di trasformazioni `albumentations`.
#    c. Salva l'immagine trasformata (aumentata).
# 6. L'utente può quindi ispezionare la cartella di output per confrontare
#    le immagini "original" e "augmented" e valutare se gli effetti sono
#    desiderati e realistici.
# --------------------------------------------------------------------------------

import os
from pathlib import Path
import numpy as np
import cv2
import albumentations as A
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# --- 1. DEFINIZIONE DELLA PIPELINE DI AUGMENTATION ---
# IMPORTANTE: Questa pipeline deve essere identica a quella utilizzata nello
# script che genera il dataset di training aumentato.
transform = A.Compose([
    # Applica una rotazione casuale entro +/- 5 gradi.
    # `border_mode=cv2.BORDER_CONSTANT` riempie i bordi con un colore nero.
    A.Rotate(limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    
    # Applica una leggera trasformazione di prospettiva.
    A.Perspective(scale=(0.01, 0.05), p=0.5),
    
    # Simula la compressione JPEG con qualità variabile.
    A.ImageCompression(quality_lower=70, quality_upper=95, p=0.5),
    
    # Aggiunge rumore gaussiano.
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    
    # Applica una sfocatura gaussiana.
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    
    # Varia casualmente luminosità e contrasto.
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
])

# --- 2. CARICAMENTO DEL DATASET ORIGINALE ---
print("Caricamento del dataset di immagini originale...")
# Specifica la directory contenente le immagini sorgenti.
image_dir = Path("../progetto_donut/images")
# Usa `load_dataset` di Hugging Face in modalità "imagefolder" per caricare le immagini.
original_dataset = load_dataset("imagefolder", data_dir=image_dir, split="train")

# --- 3. CREAZIONE DELLA DIRECTORY DI OUTPUT ---
output_dir = Path("./augmentation_samples")
output_dir.mkdir(exist_ok=True) # `exist_ok=True` evita errori se la cartella esiste già.
print(f"Salvataggio degli esempi di augmentation in: {output_dir}")

# --- 4. GENERAZIONE E SALVATAGGIO DEGLI ESEMPI ---
NUM_SAMPLES = 20 # Numero di esempi da generare.
# `tqdm` crea una barra di progresso per monitorare l'esecuzione del loop.
for i in tqdm(range(NUM_SAMPLES), desc="Generazione Esempi"):
    # Estrae un'immagine dal dataset.
    sample = original_dataset[i]
    image_pil = sample['image'].convert("RGB") # Converte in formato PIL (RGB).
    
    # Salva l'immagine originale per un confronto diretto.
    # Il formato del nome file (es. 001, 002) aiuta a mantenere l'ordine.
    image_pil.save(output_dir / f"{i:03d}_original.jpg")
    
    # Converte l'immagine PIL in un array NumPy, formato richiesto da albumentations.
    image_np = np.array(image_pil)
    
    # Applica la pipeline di trasformazioni all'array NumPy.
    augmented = transform(image=image_np)
    
    # Riconverte l'array NumPy aumentato in un'immagine PIL per poterla salvare.
    augmented_pil = Image.fromarray(augmented['image'])
    
    # Salva l'immagine aumentata.
    augmented_pil.save(output_dir / f"{i:03d}_augmented.jpg")

print(f"\nOperazione completata! Controlla le immagini nella cartella '{output_dir}'.")