# Fine-Tuning di Donut per l'Estrazione di Dati da Scontrini

Questa repository fornisce una pipeline completa, end-to-end, per il fine-tuning del modello **Donut** (Document Understanding Transformer) su un dataset personalizzato di scontrini fiscali. L'obiettivo è estrarre in modo accurato informazioni strutturate (coppie chiave-valore) dalle immagini degli scontrini, come il totale, la data, la partita IVA, ecc.

Il progetto include script per la validazione dei dati, la creazione di dataset con data augmentation, un training avanzato con callback personalizzati e un'inferenza flessibile.

## Indice
- [Funzionalità](#funzionalità)
- [Struttura dei File](#struttura-dei-file)
- [Setup e Installazione (con Conda)](#setup-e-installazione-con-conda)
- [Flusso di Lavoro](#flusso-di-lavoro)
  - [Passo 1: Prepara i Tuoi Dati](#passo-1-prepara-i-tuoi-dati)
  - [Passo 2: Valida il Ground Truth (Opzionale)](#passo-2-valida-il-ground-truth-opzionale)
  - [Passo 3: Costruisci il Dataset](#passo-3-costruisci-il-dataset)
  - [Passo 4: Verifica il Dataset Processato (Opzionale)](#passo-4-verifica-il-dataset-processato-opzionale)
  - [Passo 5: Addestra il Modello](#passo-5-addestra-il-modello)
  - [Passo 6: Esegui l'Inferenza](#passo-6-esegui-linferenza)
- [Personalizzazione](#personalizzazione)
- [Ringraziamenti](#ringraziamenti)

## Funzionalità

- **Pipeline End-to-End**: Copre ogni fase, dalla preparazione dei dati all'inferenza del modello.
- **Data Augmentation**: Utilizza la libreria `albumentations` per applicare trasformazioni realistiche alle immagini (rotazione, distorsione prospettica, rumore, sfocatura), rendendo il modello più robusto.
- **Strumenti di Validazione Dati**: Include script interattivi per validare e correggere i file JSON di ground truth e per ispezionare visivamente gli effetti della data augmentation.
- **Training Avanzato**: Sfrutta il `Seq2SeqTrainer` di Hugging Face con callback personalizzati per:
    - Salvare il processore insieme a ogni checkpoint del modello.
    - Salvare automaticamente una copia "live" del modello con le migliori performance durante il training, per consentire test immediati.
- **Processamento Efficiente**: Utilizza `torch.bfloat16` per il training a precisione mista, accelerando il processo e riducendo l'uso di memoria.
- **Inferenza Flessibile**: Fornisce uno script di inferenza che accetta argomenti da riga di comando per testare facilmente diversi modelli e immagini.

## Struttura dei File

Di seguito una panoramica degli script principali del repository:

```
.
├── images/                     # Cartella per le immagini degli scontrini (es. 001.jpg)
├── gdt/                        # Cartella per i file JSON di ground truth (es. 001.json)
│
├── requirements.txt            # Dipendenze del progetto.
│
├── json_check.py               # Tool interattivo da riga di comando per validare e correggere i file JSON.
├── build_dataset_augmented.py  # Script principale per creare i dataset di training/validazione con augmentation.
├── build_dataset.py            # Versione base dello script di creazione dataset, senza augmentation.
│
├── check_augmentations.py      # Utility per ispezionare visivamente i risultati della pipeline di augmentation.
├── check_dataset.py            # Utility per diagnosticare e verificare il formato del dataset processato.
│
├── train.py                    # Script principale per il fine-tuning del modello Donut.
│
├── inference_checkpoints.py    # Script flessibile per eseguire l'inferenza tramite argomenti da riga di comando.
├── inference.py                # Versione base e hardcoded dello script di inferenza.
└── README.md                   # Questo file.
```

## Setup e Installazione (con Conda)

1.  **Clona il repository:**
    ```bash
    git clone https://github.com/tuo-username/donut-receipt-parser.git
    cd donut-receipt-parser
    ```

2.  **Crea e attiva un ambiente Conda:**
    Si consiglia di usare Python 3.10 o una versione successiva.
    ```bash
    conda create --name donut-env python=3.10
    conda activate donut-env
    ```

3.  **Installa PyTorch (Passo Fondamentale):**
    Per garantire la massima compatibilità con la tua GPU, è fortemente consigliato installare PyTorch tramite Conda. Visita il [sito ufficiale di PyTorch](https://pytorch.org/get-started/locally/) per ottenere il comando esatto per la tua configurazione (OS, versione di CUDA, ecc.).

    *Esempio di comando per CUDA 12.1 (verifica quello corretto per il tuo sistema!):*
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4.  **Installa le altre dipendenze:**
    Una volta installato PyTorch, puoi installare le restanti librerie usando `pip` e il file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Flusso di Lavoro

Segui questi passaggi per preparare i dati, addestrare il modello ed eseguire l'inferenza.

### Passo 1: Prepara i Tuoi Dati

1.  Inserisci tutte le immagini degli scontrini nella cartella `images/`.
2.  Crea un file JSON corrispondente per ogni immagine nella cartella `gdt/`. Il nome del file JSON deve coincidere con quello dell'immagine (es. `images/001.jpg` e `gdt/001.json`).

Il file JSON deve contenere le coppie chiave-valore che vuoi estrarre. Esempio:
```json
{
  "total": "77.00",
  "date": "2024-12-05",
  "vat_number": "03920700246",
  "company": "RISTORANTE LA CUPOLA"
  "currency": "EUR"
  "address": null
}
```

### Passo 2: Valida il Ground Truth (Opzionale)

Per assicurarti che i tuoi file JSON siano corretti, esegui lo script di validazione interattiva. Analizzerà tutti i file nella cartella `gdt/` e ti chiederà di correggere eventuali errori (es. virgole nei totali, formati di data non validi).

```bash
python json_check.py
```

### Passo 3: Costruisci il Dataset

Questo script elabora le tue immagini e i file JSON, applica la data augmentation allo split di training e salva su disco i dataset pronti all'uso e il processore.

```bash
python build_dataset_augmented.py
```
Questo comando creerà quattro nuove cartelle:
- `processed_dataset_train_aug/`: Il dataset di training.
- `processed_dataset_validation_aug/`: Il dataset di validazione.
- `donut_processor_finetuned_aug/`: Il processore Donut configurato.
- `augmentation_samples/` (creata da `check_augmentations.py`): Immagini di esempio per verificare gli effetti dell'augmentation.

### Passo 4: Verifica il Dataset Processato (Opzionale)

Prima di avviare un lungo ciclo di training, è consigliabile verificare che il dataset sia stato costruito correttamente. Questo script ispeziona un campione e ne controlla la forma, i tipi di dato e la tokenizzazione.

```bash
python check_dataset.py
```
*(Nota: Potrebbe essere necessario modificare i percorsi all'interno dello script se stai verificando il dataset aumentato).*

### Passo 5: Addestra il Modello

Avvia il processo di fine-tuning con lo script `train.py`. Caricherà i dataset processati e inizierà l'addestramento. I checkpoint verranno salvati nella cartella `donut-finetuned-receipts/`.

```bash
python train.py
```
Durante il training, una copia del modello con le migliori performance verrà salvata in `donut-finetuned-receipts/best_model_live/`, permettendoti di testarlo senza interrompere il processo. Al termine, il modello finale migliore sarà salvato in `donut-finetuned-receipts/best_model_final/`.

### Passo 6: Esegui l'Inferenza

Usa lo script `inference_checkpoints.py` per estrarre informazioni da una nuova immagine. Devi fornire il percorso al modello addestrato e all'immagine.

```bash
python inference_checkpoints.py --model_path ./donut-finetuned-receipts/best_model_final --image_path ./images/000.jpg
```
Lo script stamperà i dati estratti in formato JSON sulla console.

## Personalizzazione

-   **Iperparametri**: Tutte le impostazioni di training (learning rate, numero di epoche, batch size, ecc.) possono essere modificate nell'oggetto `Seq2SeqTrainingArguments` all'interno di `train.py`.
-   **Data Augmentation**: La pipeline di augmentation è definita all'inizio di `build_dataset_augmented.py`. Puoi aggiungere, rimuovere o modificare le trasformazioni di `albumentations` per adattarle meglio al tuo dataset. Usa `check_augmentations.py` per visualizzare le modifiche.
-   **Modello e Dimensioni Immagine**: Il modello base e le dimensioni di input delle immagini sono impostate in `build_dataset_augmented.py`. Se le modifichi, assicurati di usare la stessa configurazione in `train.py`.

## Ringraziamenti
*   Questo progetto si basa sul modello **Donut**, introdotto nell'articolo [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) di Geewook Kim et al.
*   L'implementazione sfrutta ampiamente le librerie `transformers` e `datasets` di [Hugging Face](https://huggingface.co/).
