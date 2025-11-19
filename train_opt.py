# train.py (Versione Aggiornata e Ottimizzata)
# --------------------------------------------------------------------------------
# DESCRIZIONE:
# Questo script gestisce il fine-tuning di un modello Donut per l'estrazione
# di informazioni da scontrini.
#
# AGGIORNAMENTI IN QUESTA VERSIONE:
# 1. È stata implementata una funzione `compute_metrics` per calcolare
#    Precisione, Recall e F1-score, offrendo una valutazione più accurata.
# 2. È stato introdotto l'Early Stopping per fermare il training al momento
#    ottimale, prevenendo l'overfitting e risparmiando risorse.
# 3. Il criterio per la selezione del miglior modello è ora basato sull'F1-score
#    invece che sulla semplice loss di validazione.
# --------------------------------------------------------------------------------

import os
import re
import shutil
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    DonutProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    TrainerCallback,
    EvalPrediction,  # Oggetto usato dalla funzione compute_metrics
)
import warnings

# --- CONFIGURAZIONE INIZIALE ---
warnings.filterwarnings("ignore", category=FutureWarning)
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# --- CALLBACK PERSONALIZZATI ( invariati ) ---
class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_dir):
            self.processor.save_pretrained(checkpoint_dir)
            print(f"\n[Callback] Processor salvato in: {checkpoint_dir}")
        else:
            print(f"\n[Callback] Attenzione: la cartella {checkpoint_dir} non esiste. Salto il salvataggio del processore.")


class LiveBestModelCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_world_process_zero and state.best_model_checkpoint and metrics:
            current_metric = metrics.get("eval_" + args.metric_for_best_model) # 'eval_f1'
            if state.best_metric is not None and current_metric == state.best_metric:
                best_checkpoint_path = state.best_model_checkpoint
                live_best_model_dir = os.path.join(args.output_dir, "best_model_live")
                print(f"\n--- [Callback] Nuovo miglior modello trovato! {args.metric_for_best_model}: {state.best_metric:.4f} ---")
                print(f"--- Copiando {best_checkpoint_path} in {live_best_model_dir} ---")
                if os.path.exists(live_best_model_dir):
                    shutil.rmtree(live_best_model_dir)
                shutil.copytree(best_checkpoint_path, live_best_model_dir)
                self.processor.save_pretrained(live_best_model_dir)
                print("--- [Callback] Aggiornamento del miglior modello 'live' completato. ---\n")


# --- 1. CARICAMENTO DATI E PROCESSORE ---
print("Caricamento del dataset di training, validazione e del processore...")
try:
    train_dataset = load_from_disk("./processed_dataset_train_aug")
    eval_dataset = load_from_disk("./processed_dataset_validation_aug")
    processor = DonutProcessor.from_pretrained("./donut_processor_finetuned_aug")
except FileNotFoundError:
    print("ERRORE: Uno o più file del dataset non sono stati trovati. Esegui prima lo script di preparazione.")
    exit()
print("Dati caricati con successo.")


# --- 2. CONFIGURAZIONE DEL MODELLO (invariata) ---
print("\nCaricamento del modello base da Hugging Face Hub...")
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base", torch_dtype=torch.bfloat16
)
print("Configurazione del modello per il fine-tuning...")
model.decoder.resize_token_embeddings(len(processor.tokenizer))
model.config.encoder.image_size = [processor.image_processor.size["height"], processor.image_processor.size["width"]]
max_len = max(len(seq) for seq in train_dataset["labels"])
model.config.decoder.max_length = max_len
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4


# --- NUOVA SEZIONE: FUNZIONE PER CALCOLARE LE METRICHE DI PERFORMANCE ---
def compute_metrics(p: EvalPrediction):
    """
    Calcola Precisione, Recall e F1-score a livello di campo confrontando
    le predizioni del modello con le etichette di riferimento.
    """
    # Decodifica le predizioni e le etichette da ID di token a stringhe di testo.
    predictions = processor.batch_decode(p.predictions, skip_special_tokens=True)
    
    # Prepara le etichette, sostituendo -100 (token ignorati dalla loss) con il pad_token_id.
    labels = np.where(p.label_ids != -100, p.label_ids, processor.tokenizer.pad_token_id)
    labels = processor.batch_decode(labels, skip_special_tokens=True)

    true_positives, false_positives, false_negatives = 0, 0, 0

    def parse_donut_string(text: str) -> set:
        """Estrae coppie (chiave, valore) da una stringa formato Donut."""
        # Usa una regex per trovare tutte le coppie <s_chiave>valore</s_chiave>
        pattern = r"<s_(\w+)>(.*?)</s_\1>"
        return set(re.findall(pattern, text))

    # Itera su ogni coppia di predizione/etichetta nel batch di validazione
    for pred_str, label_str in zip(predictions, labels):
        # Estrai le coppie (chiave, valore) come set per un confronto efficiente
        pred_items = parse_donut_string(pred_str)
        label_items = parse_donut_string(label_str)

        # Calcola TP, FP, FN usando le operazioni sui set
        true_positives += len(pred_items.intersection(label_items))
        false_positives += len(pred_items.difference(label_items))
        false_negatives += len(label_items.difference(pred_items))
    
    # Calcola le metriche finali, gestendo la divisione per zero
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # La funzione deve restituire un dizionario di metriche
    return {"precision": precision, "recall": recall, "f1": f1}


# --- 3. IMPOSTAZIONE DEGLI ARGOMENTI DI TRAINING (AGGIORNATI) ---
output_dir = "./donut-finetuned-receipts"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=40,                 # Ora è un limite massimo, il training si fermerà prima grazie all'early stopping.
    learning_rate=2e-5,                  # Valore standard, si può provare 1e-5 per una convergenza più fine.
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    bf16=True,
    dataloader_num_workers=4,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=250,
    evaluation_strategy="steps",
    eval_steps=250,
    save_total_limit=3,
    
    # --- MODIFICHE CHIAVE ---
    load_best_model_at_end=True,         # Carica il miglior modello alla fine del training.
    metric_for_best_model="f1",          # Seleziona il modello migliore in base all'F1-score.
    greater_is_better=True,              # Un F1-score più alto è migliore.
    
    # --- ABILITAZIONE EARLY STOPPING ---
    # Ferma il training se la metrica `f1` non migliora per 5 valutazioni consecutive.
    # Questo previene l'overfitting e conclude il training al momento ottimale.
    early_stopping_patience=5,
    early_stopping_threshold=0.001,      # Un nuovo "miglior" modello deve avere un miglioramento di almeno 0.001.

    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    predict_with_generate=True,          # NECESSARIO per la valutazione basata sulla generazione di testo.
)


# --- 4. CREAZIONE E AVVIO DEL TRAINER (AGGIORNATO) ---
save_processor_cb = SaveProcessorCallback(processor)
live_best_model_cb = LiveBestModelCallback(processor)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[save_processor_cb, live_best_model_cb],
    compute_metrics=compute_metrics,  # Passa la nuova funzione per le metriche.
)

print("\nConfigurazione completata. Inizio del training ottimizzato con Early Stopping e F1-score...")
trainer.train(resume_from_checkpoint=False)

print("\nTraining completato (o fermato da Early Stopping).")
print("Grazie a 'load_best_model_at_end=True', il miglior modello è stato caricato automaticamente.")

# --- SALVATAGGIO FINALE DEL MIGLIOR MODELLO ---
best_model_dir = os.path.join(output_dir, "best_model_final")
os.makedirs(best_model_dir, exist_ok=True)
print(f"\nSalvataggio del miglior modello finale e del processore in: {best_model_dir}")
trainer.save_model(best_model_dir)
processor.save_pretrained(best_model_dir)

print("\nOperazione terminata con successo!")