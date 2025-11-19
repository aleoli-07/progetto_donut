# json_check.py
# --------------------------------------------------------------------------------
# DESCRIZIONE:
# Questo script fornisce una utility interattiva da riga di comando per validare
# e correggere file JSON all'interno di una directory specificata. È progettato
# per garantire la coerenza e la correttezza dei dati di ground truth prima
# di utilizzarli per il training di un modello.
#
# FUNZIONAMENTO:
# 1. Scansiona una directory alla ricerca di file con estensione `.json`.
# 2. Per ogni file, esegue una serie di controlli di validazione specifici:
#    - Formato della valuta (`currency`): deve essere un codice ISO di 3 lettere.
#    - Formato del totale (`total`): deve essere un numero e usare il punto come
#      separatore decimale.
#    - Formato della data (`date`): deve essere nel formato `YYYY-MM-DD`.
#    - Tipo del `vat_number`: deve essere una stringa.
# 3. Se vengono trovati errori, li presenta all'utente in modo dettagliato,
#    mostrando il valore errato, il motivo dell'errore e un suggerimento.
# 4. Chiede all'utente se desidera correggere l'errore, saltare il file o uscire.
# 5. Se l'utente sceglie di correggere, può inserire un nuovo valore.
# 6. Al termine dell'analisi di un file, se sono state apportate modifiche,
#    chiede conferma prima di sovrascrivere il file originale.
# --------------------------------------------------------------------------------

import json
import re
import os
from datetime import datetime
from pathlib import Path

def validate_json_data(json_data):
    """
    Valida un singolo dizionario JSON rispetto a una serie di regole predefinite.
    Ignora i campi non presenti o con valore `None`.

    Args:
        json_data (dict): Il dizionario JSON da validare.

    Returns:
        list: Una lista di dizionari, dove ogni dizionario rappresenta un errore
              rilevato. La lista è vuota se non ci sono errori.
    """
    errors = []

    # 1. Controllo `currency`: deve essere un codice ISO di 3 lettere maiuscole (es. "EUR").
    if 'currency' in json_data and json_data['currency'] is not None:
        currency = json_data['currency']
        if not isinstance(currency, str) or not re.fullmatch(r'^[A-Z]{3}$', currency):
            errors.append({
                "field": "currency",
                "current_value": currency,
                "error_message": f"Il campo 'currency' '{currency}' non è nel formato ISO standard.",
                "suggestion": "Correggi in un codice ISO di 3 lettere maiuscole (es. 'EUR', 'USD')."
            })

    # 2. Controllo `total`: deve essere un numero valido e non usare la virgola.
    if 'total' in json_data and json_data['total'] is not None:
        total = json_data['total']
        if isinstance(total, str):
            # Controlla l'uso della virgola come separatore decimale.
            if ',' in total:
                errors.append({
                    "field": "total",
                    "current_value": total,
                    "error_message": f"Il campo 'total' '{total}' usa la virgola ',' invece del punto '.'.",
                    "suggestion": f"Sostituisci la virgola con il punto (es. '{total.replace(',', '.')}') o converti in numero."
                })
            # Controlla se la stringa può essere convertita in un float.
            try:
                float(total.replace(',', '.'))
            except (ValueError, TypeError):
                errors.append({
                    "field": "total",
                    "current_value": total,
                    "error_message": f"Il campo 'total' '{total}' non è una stringa numerica valida.",
                    "suggestion": "Assicurati che sia un numero valido (es. '123.45')."
                })
        # Se non è una stringa, controlla che sia un int o un float.
        elif not isinstance(total, (int, float)):
             errors.append({
                "field": "total",
                "current_value": total,
                "error_message": f"Il campo 'total' '{total}' non è un tipo numerico valido.",
                "suggestion": "Assicurati che sia un numero (intero o float)."
            })

    # 3. Controllo `date`: deve essere una stringa nel formato YYYY-MM-DD.
    if 'date' in json_data and json_data['date'] is not None:
        date_str = json_data['date']
        if not isinstance(date_str, str):
            errors.append({
                "field": "date",
                "current_value": date_str,
                "error_message": f"Il campo 'date' non è una stringa.",
                "suggestion": "Converti il valore in una stringa nel formato YYYY-MM-DD."
            })
        # Procede solo se la stringa non è vuota.
        elif date_str.strip() != "":
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                errors.append({
                    "field": "date",
                    "current_value": date_str,
                    "error_message": f"Il campo 'date' '{date_str}' non è nel formato YYYY-MM-DD.",
                    "suggestion": "Correggi il formato della data (es. '2023-10-26')."
                })

    # 4. Controllo `vat_number`: deve essere una stringa.
    if 'vat_number' in json_data and json_data['vat_number'] is not None:
        vat_number = json_data['vat_number']
        if not isinstance(vat_number, str):
            errors.append({
                "field": "vat_number",
                "current_value": vat_number,
                "error_message": f"Il campo 'vat_number' non è una stringa.",
                "suggestion": "Converti il valore in una stringa."
            })

    return errors

def save_json_file(filepath, data):
    """
    Salva un dizionario in un file JSON con formattazione leggibile.

    Args:
        filepath (str): Il percorso del file in cui salvare i dati.
        data (dict): Il dizionario da salvare.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"File '{filepath}' salvato con successo.")
    except Exception as e:
        print(f"Errore durante il salvataggio del file '{filepath}': {e}")


def interactive_json_validation(directory_path):
    """
    Funzione principale che orchestra il processo di validazione interattiva.
    Scansiona una directory, valida ogni file JSON e guida l'utente nella correzione.
    
    Args:
        directory_path (str or Path): Il percorso della directory contenente i file JSON.
    """
    if not os.path.isdir(directory_path):
        print(f"Errore: La directory '{directory_path}' non esiste.")
        return

    print(f"\n--- Inizio Validazione Interattiva in: {directory_path} ---\n")

    # `sorted` garantisce un ordine di elaborazione consistente e ripetibile.
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            print(f"\n--- Elaborazione file: {filepath} ---")

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  Errore di parsing: Il file non è un JSON valido. Dettagli: {e}")
                continue # Passa al file successivo.

            # Crea una copia dell'originale per confrontare le modifiche alla fine.
            original_json_data = json_data.copy()
            errors = validate_json_data(json_data)
            
            if not errors:
                print("  Nessun errore rilevato in questo file.")
                continue

            print("\n  Errori rilevati:")
            for i, error in enumerate(errors):
                # Mostra l'errore in dettaglio.
                print(f"\n  {i+1}. Campo: '{error['field']}'")
                print(f"     Valore attuale: '{error['current_value']}'")
                print(f"     Messaggio: {error['error_message']}")
                print(f"     Suggerimento: {error['suggestion']}")

                # Chiede all'utente l'azione da intraprendere.
                action = input("     Azione (s=correggi, n=ignora, salta_file, esci): ").lower()

                if action == 's':
                    new_value_input = input(f"       Inserisci il nuovo valore per '{error['field']}' (invio per annullare): ")
                    
                    if new_value_input.strip() == "":
                        print("       Nessuna modifica applicata.")
                        continue

                    # Aggiorna temporaneamente il dizionario con il nuovo valore.
                    json_data[error['field']] = new_value_input
                    print(f"       Campo '{error['field']}' aggiornato a: '{json_data[error['field']]}'")

                elif action == 'salta_file':
                    print("  Saltando il resto degli errori per questo file...")
                    break # Interrompe il loop degli errori e passa al file successivo.
                elif action == 'esci':
                    print("  Uscita dal programma.")
                    return # Termina l'esecuzione dello script.
                else:
                    print("  Errore ignorato.")

            # Dopo aver analizzato tutti gli errori di un file, controlla se sono state fatte modifiche.
            if json_data != original_json_data:
                save_action = input(f"\n  Salvare le modifiche a '{filepath}'? (s/n): ").lower()
                if save_action == 's':
                    save_json_file(filepath, json_data)
                else:
                    print(f"  Modifiche scartate per '{filepath}'.")
            else:
                print("  Nessuna modifica apportata a questo file.")

    print("\n--- Validazione Interattiva Completata ---")


# --- BLOCCO DI ESECUZIONE PRINCIPALE ---
if __name__ == "__main__":
    # Imposta qui il percorso della directory da controllare.
    # L'uso di `pathlib.Path` è una buona pratica per gestire i percorsi in modo cross-platform.
    my_directory_path = Path("./gdt")

    # Avvia la funzione di validazione interattiva.
    interactive_json_validation(my_directory_path)