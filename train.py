# train.py
# --------------------------------------------------------------------------------
# DESCRIZIONE:
# Questo script gestisce il processo di fine-tuning (fine-tuning) di un modello
# VisionEncoderDecoderModel (specificamente Donut) per il task di estrazione
# di informazioni da scontrini.
#
# FUNZIONAMENTO:
# 1. Carica un dataset pre-processato e un processore Donut.
# 2. Inizializza un modello Donut pre-addestrato da Hugging Face.
# 3. Configura gli argomenti di training (iperparametri, percorsi, strategie).
# 4. Implementa due callback personalizzati con `Seq2SeqTrainer`:
#    - `SaveProcessorCallback`: Salva il processore insieme a ogni checkpoint
#      del modello, rendendo ogni checkpoint autonomo e testabile.
#    - `LiveBestModelCallback`: Identifica il miglior modello durante il training
#      basandosi sulla metrica di validazione (loss) e ne salva una copia "live"
#      in una cartella separata (`best_model_live`), permettendo di testare
#      il miglior modello trovato fino a quel momento senza attendere la fine.
# 5. Avvia il training e, al termine, salva il miglior modello finale.
# --------------------------------------------------------------------------------

import os
import shutil
import torch
from datasets import load_from_disk
from transformers import (
    DonutProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    TrainerCallback,
)
import warnings

# --- CONFIGURAZIONE INIZIALE ---

# Ignora i FutureWarning per mantenere l'output pulito da messaggi non essenziali.
warnings.filterwarnings("ignore", category=FutureWarning)

# Se è disponibile una GPU, svuota la cache per liberare memoria VRAM.
# Utile per evitare errori "out-of-memory" all'avvio.
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# --- CALLBACK 1: SALVATAGGIO DEL PROCESSORE CON OGNI CHECKPOINT ---
# ---------------------------------------------------------------------
class SaveProcessorCallback(TrainerCallback):
    """
    Callback personalizzato che salva il `DonutProcessor` all'interno di ogni
    cartella di checkpoint creata dal Trainer. Questo garantisce che ogni
    checkpoint sia un artefatto completo e riutilizzabile, contenente sia i pesi
    del modello sia il processore necessario per l'inferenza.
    """
    def __init__(self, processor):
        """
        Inizializza il callback.
        Args:
            processor (DonutProcessor): L'istanza del processore da salvare.
        """
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        """
        Metodo eseguito automaticamente dal Trainer ogni volta che viene salvato
        un checkpoint (triggerato da `save_strategy` e `save_steps`).
        """
        # Costruisce il percorso della directory del checkpoint corrente in modo robusto,
        # basandosi sullo stato globale del training (numero di step).
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        # Verifica che la directory del checkpoint esista prima di tentare il salvataggio.
        if os.path.exists(checkpoint_dir):
            self.processor.save_pretrained(checkpoint_dir)
            print(f"\n[Callback] Processor salvato in: {checkpoint_dir}")
        else:
            # Messaggio di avviso nel caso improbabile che la cartella non sia stata creata.
            print(f"\n[Callback] Attenzione: la cartella {checkpoint_dir} non esiste. Salto il salvataggio del processore.")


# ---------------------------------------------------------------------
# --- CALLBACK 2: SALVATAGGIO "LIVE" DEL MIGLIOR MODELLO ---
# ---------------------------------------------------------------------
class LiveBestModelCallback(TrainerCallback):
    """
    Callback che, dopo ogni ciclo di valutazione, verifica se il modello corrente
    è il migliore trovato finora. In caso affermativo, copia l'intero checkpoint
    del miglior modello in una directory separata (`best_model_live`).
    Questo permette di avere sempre a disposizione il miglior modello "live"
    per test o inferenze parallele, senza dover attendere la fine del training.
    """
    def __init__(self, processor):
        """
        Inizializza il callback.
        Args:
            processor (DonutProcessor): L'istanza del processore da salvare insieme al modello.
        """
        self.processor = processor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Metodo eseguito automaticamente dal Trainer dopo ogni valutazione
        (triggerato da `evaluation_strategy` e `eval_steps`).
        """
        # Controlla se è stato trovato un nuovo miglior checkpoint.
        # `state.best_model_checkpoint` viene aggiornato internamente dal Trainer.
        if state.is_world_process_zero and state.best_model_checkpoint and metrics:
            
            # Recupera il valore della metrica corrente (es. 'loss').
            current_metric = metrics.get(args.metric_for_best_model)
            
            # Se la metrica corrente corrisponde alla migliore metrica registrata,
            # significa che questo è il checkpoint che ha stabilito il nuovo record.
            if state.best_metric is not None and current_metric == state.best_metric:
                best_checkpoint_path = state.best_model_checkpoint
                live_best_model_dir = os.path.join(args.output_dir, "best_model_live")

                print(f"\n--- [Callback] Nuovo miglior modello trovato! Loss: {state.best_metric:.4f} ---")
                print(f"--- Copiando {best_checkpoint_path} in {live_best_model_dir} ---")

                # Rimuove la vecchia versione del miglior modello live per sostituirla.
                if os.path.exists(live_best_model_dir):
                    shutil.rmtree(live_best_model_dir)
                
                # Copia l'intera cartella del checkpoint nella destinazione "live".
                shutil.copytree(best_checkpoint_path, live_best_model_dir)
                
                # Salva anche il processore, rendendo la cartella "live" completa.
                self.processor.save_pretrained(live_best_model_dir)
                print("--- [Callback] Aggiornamento del miglior modello 'live' completato. ---\n")

# --- 1. CARICAMENTO DATI E PROCESSORE ---
print("Caricamento del dataset di training, validazione e del processore...")
try:
    # Carica i dataset pre-processati dal disco.
    train_dataset = load_from_disk("./processed_dataset_train_aug")
    eval_dataset = load_from_disk("./processed_dataset_validation_aug")
    # Carica il processore (che contiene tokenizer e image_processor) sintonizzato sul nostro vocabolario.
    processor = DonutProcessor.from_pretrained("./donut_processor_finetuned_aug")
except FileNotFoundError:
    print("ERRORE: Uno o più file del dataset non sono stati trovati. Esegui prima lo script di preparazione.")
    exit()
print("Dati caricati con successo.")


# --- 2. CONFIGURAZIONE DEL MODELLO ---
print("\nCaricamento del modello base da Hugging Face Hub...")
# Carica il modello pre-addestrato "donut-base".
# `torch_dtype=torch.bfloat16` abilita il mixed-precision training per ridurre l'uso di memoria
# e accelerare il calcolo su hardware compatibile (es. GPU Ampere).
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base", torch_dtype=torch.bfloat16
)

print("Configurazione del modello per il fine-tuning...")
# Adatta la dimensione del vocabolario del decoder a quella del nostro tokenizer personalizzato.
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# Imposta la dimensione delle immagini attesa dall'encoder.
model.config.encoder.image_size = [processor.image_processor.size["height"], processor.image_processor.size["width"]]

# Calcola la lunghezza massima delle sequenze nel dataset per impostare un limite ragionevole.
max_len = max(len(seq) for seq in train_dataset["labels"])
model.config.decoder.max_length = max_len

# Imposta i token speciali nella configurazione del modello.
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
# Imposta il token di inizio sequenza per la generazione del decoder.
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")

# Parametri per la generazione (inference-time), usati durante la valutazione.
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4


# --- 3. IMPOSTAZIONE DEGLI ARGOMENTI DI TRAINING ---
output_dir = "./donut-finetuned-receipts"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=40,
    learning_rate=2e-5,
    per_device_train_batch_size=2,       # Batch size per GPU/CPU durante il training.
    per_device_eval_batch_size=4,        # Batch size per GPU/CPU durante la validazione.
    gradient_accumulation_steps=8,       # Accumula i gradienti per N step prima di un aggiornamento. Simula un batch size più grande (2 * 8 = 16).
    bf16=True,                           # Abilita il training con BFloat16.
    dataloader_num_workers=4,            # Numero di thread per caricare i dati.
    weight_decay=0.01,                   # Regularizzazione L2.
    logging_strategy="steps",            # Logga le metriche ogni N step.
    logging_steps=50,
    save_strategy="steps",               # Salva i checkpoint ogni N step.
    save_steps=250,
    evaluation_strategy="steps",         # Esegui la validazione ogni N step.
    eval_steps=250,
    save_total_limit=3,                  # Mantieni solo gli ultimi 3 checkpoint per risparmiare spazio.
    load_best_model_at_end=True,         # Al termine del training, carica automaticamente i pesi del miglior modello.
    metric_for_best_model="loss",        # Metrica usata per determinare il "miglior" modello.
    greater_is_better=False,             # La metrica "loss" è migliore quando è più bassa.
    lr_scheduler_type="cosine",          # Tipo di learning rate scheduler.
    warmup_ratio=0.1,                    # Percentuale di step iniziali con learning rate crescente (warmup).
    predict_with_generate=True,          # Usa `model.generate()` per la valutazione, necessario per metriche basate sulla generazione di testo.
)


# --- 4. CREAZIONE E AVVIO DEL TRAINER ---
# Istanzia i callback personalizzati.
save_processor_cb = SaveProcessorCallback(processor)
live_best_model_cb = LiveBestModelCallback(processor)

# Crea l'oggetto `Seq2SeqTrainer` con modello, argomenti, dataset e callback.
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[save_processor_cb, live_best_model_cb] # Passa la lista dei callback da eseguire.
)

print("\nConfigurazione completata. Inizio del training ottimizzato...")
# Avvia il processo di training. `resume_from_checkpoint=False` inizia un nuovo training.
# Impostarlo a `True` riprenderebbe dall'ultimo checkpoint se presente.
trainer.train(resume_from_checkpoint=False)

print("\nTraining completato.")
print("Grazie a 'load_best_model_at_end=True', il miglior modello è stato caricato automaticamente.")

# --- SALVATAGGIO FINALE DEL MIGLIOR MODELLO ---
# Sebbene il callback "live" salvi il miglior modello, è buona pratica
# creare una copia finale "ufficiale" al termine del training.
best_model_dir = os.path.join(output_dir, "best_model_final")
os.makedirs(best_model_dir, exist_ok=True)
print(f"\nSalvataggio del miglior modello finale e del processore in: {best_model_dir}")
trainer.save_model(best_model_dir)
processor.save_pretrained(best_model_dir)

print("\nOperazione terminata con successo!")