import warnings
warnings.filterwarnings("ignore")
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdhxvYTJ9QtcJWSe4Ls6FFqA2CwSiaF8OPJpqw7nnuU99qsZw/viewform?usp=dialog"

print("=" * 50)
print("  QUESTIONARIO GOOGLE MODULI")
print("  Il browser si aprirà automaticamente.")
print("  La pagina si chiuderà da sola dopo l'invio.")
print("=" * 50)

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

try:
    print("\n[INFO] Apertura del questionario...")
    driver.get(FORM_URL)
    
    print("[INFO] In attesa della compilazione...")
    WebDriverWait(driver, 300).until(EC.url_contains('formResponse'))
    
    print("\n[INFO] Questionario inviato! Chiusura automatica tra 3 secondi...")
    time.sleep(2)
    
except Exception as e:
    print(f"\n[ERRORE] Si è verificato un problema: {e}")
finally:
    driver.quit()
    print("Programma terminato.")
