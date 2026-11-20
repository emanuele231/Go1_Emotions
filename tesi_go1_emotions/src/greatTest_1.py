import random
import subprocess
import sys
import time
import os

CASI_DISPONIBILI = [
    "go1-case-1.py", "go1-case-2.py", "go1-case-3.py", "go1-case-4.py",
    "go1-case-5.py", "go1-case-6.py", "go1-case-7.py", "go1-case-8.py",
    "go1-case-9.py", "go1-case-10.py", "go1-case-11.py", "go1-case-12.py",
    "go1-case-13.py", "go1-case-14.py", "go1-case-15.py", "go1-case-16.py",
    "go1-case-17.py", "go1-case-18.py"
]

NUMERO_CASI_DA_ESEGUIRE = 3
PAUSA_TRA_I_CASI = 3.0 

# ============================================================
# LOGICA DI ESECUZIONE
# ============================================================
def main():
    
    try:
        casi_selezionati = random.sample(CASI_DISPONIBILI, NUMERO_CASI_DA_ESEGUIRE)
    except ValueError:
        print("[ERRORE] Non ci sono abbastanza casi disponibili nella lista.")
        sys.exit(1)

    print(f"\n[INFO] Casi selezionati per questa sessione:")
    for i, caso in enumerate(casi_selezionati, 1):
        print(f"  {i}. {caso}")
    print("-" * 60)
    
    input("Premi INVIO per iniziare l'esecuzione del primo caso...\n")

    for indice, file_caso in enumerate(casi_selezionati, 1):
        print(f"\n{'='*60}")
        print(f"  AVVIO CASO {indice}/{NUMERO_CASI_DA_ESEGUIRE}: {file_caso}")
        print(f"{'='*60}")
        
        if not os.path.exists(file_caso):
            print(f"[ERRORE CRITICO] Il file '{file_caso}' non esiste. Salto al prossimo.")
            continue

        try:
            subprocess.run([sys.executable, file_caso], check=True)
            print(f"\n[SUCCESSO] '{file_caso}' terminato correttamente.")
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERRORE] '{file_caso}' è terminato con un errore (Codice: {e.returncode}).")
        except KeyboardInterrupt:
            print("\n[INTERRUZIONE] Esecuzione interrotta manualmente dall'utente.")
            break
        except Exception as e:
            print(f"\n[ERRORE IMPREVISTO] durante l'esecuzione di '{file_caso}': {e}")
        
        if indice < NUMERO_CASI_DA_ESEGUIRE:
            print(f"\n[ATTESA] Pausa di {PAUSA_TRA_I_CASI} secondi prima del prossimo caso...")
            time.sleep(PAUSA_TRA_I_CASI)


if __name__ == "__main__":
    main()
