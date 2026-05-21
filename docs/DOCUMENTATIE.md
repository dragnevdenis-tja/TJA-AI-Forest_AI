# Forest Audio AI: Sistem Automatizat de Monitorizare a Pădurilor

Forest Audio AI este un sistem complex de deep learning conceput pentru supravegherea mediului în timp real. Acesta identifică sunete ambientale și antropice (ex: incendii de pădure, drujbe, animale sălbatice, ploaie) folosind o arhitectură hibridă de Rețea Neuronală Recurentă Convoluțională (CRNN).

## 🚀 Pornire Rapidă

### 1. Cerințe Preliminare
- **Python**: 3.9+ (Testat pe 3.13/3.14)
- **Node.js și npm**: Pentru interfața web 3D
- **FFmpeg**: Necesar pentru procesarea audio (gestionat automat via `static-ffmpeg`)

### 2. Configurare Backend
Clonați depozitul și instalați dependențele Python:
```bash
# Creați un mediu virtual
python -m venv .venv
source .venv/bin/activate  # Sau .venv\Scripts\activate pe Windows

# Instalați dependențele
pip install -r requirements.txt
```

### 3. Configurare Interfață Web
Proiectul include o interfață modernă cu glob 3D.
```bash
# Configurare Frontend
cd forest-audio-web/frontend
npm install
npm run dev
```

Într-un terminal separat, porniți backend-ul FastAPI:
```bash
# Din rădăcina proiectului
python -m uvicorn forest-audio-web.app.main:app --reload --port 8000
```

---

## 🏗️ Arhitectură și Flux de Lucru

### Modelul de Bază (CRNN)
Sistemul folosește o **Rețea Neuronală Recurentă Convoluțională**:
- **Straturi CNN**: Extrag caracteristici spațiale din Mel-spectrograme („imaginile” audio).
- **Straturi RNN (GRU)**: Captează secvențele temporale și durata sunetelor.
- **PCEN (Antrenabil)**: Normalizarea energiei pe canal pentru extragerea robustă a caracteristicilor în medii zgomotoase.

### Fluxul de Date
1.  **Colectare Automatizată**: `trainer.py` include logică integrată pentru a descărca și eticheta mostre audio de pe YouTube folosind `yt-dlp`.
2.  **Integrarea Seturilor de Date**: Suportă surse multiple, inclusiv FSC22, ESC50 și seturi de date personalizate pentru incendii de pădure.
3.  **Securizare Adversarială**: Scriptul `validator.py` antrenează un model secundar pentru a detecta potențiale clasificări greșite ale modelului principal, creând o buclă de auto-îmbunătățire.

---

## 🛠️ Componente Cheie și Scripturi

### `scripts/trainer.py`
Motorul principal de antrenare.
- **Funcționalități**: Construirea automată a manifestului, integrarea datelor YouTube, suport multi-GPU și augmentare extinsă.
- **Utilizare**: `python scripts/trainer.py --data_root ./Data --epochs 50`

### `scripts/runnner.py`
Script de inferență de înaltă performanță și evaluare în loturi.
- **Funcționalități**: Implementare PCEN, predicție la nivel de cadru și I/O audio robust via `soundfile`.

### `scripts/validator.py`
Implementează arhitectura „Validator” pentru securizarea modelului. Se concentrează pe identificarea cazurilor limită unde clasificatorul principal ar putea eșua.

### `forest-audio-web/`
O aplicație web gata de producție:
- **FastAPI Backend**: Gestionează procesarea audio în timp real și managementul nodurilor.
- **React Frontend**: Un glob 3D bazat pe Three.js unde utilizatorii pot plasa senzori virtuali și monitoriza intrarea audio de la microfon în timp real.

---

## 📦 Desfășurare (Deployment)

Proiectul este conceput pentru a fi **gata de desfășurare**:
- **Multiplatformă**: Folosește `soundfile` și `static-ffmpeg` pentru a asigura compatibilitatea audio pe Windows și Linux.
- **Cerințe**: Versiunile fixate în `requirements.txt` asigură medii de lucru consistente.
- **Checkpoints**: Cele mai bune ponderi ale modelului sunt salvate în `models/best_checkpoint.pth`.

---

## 📝 Licență
Acest proiect este destinat conservării și sistemelor de avertizare timpurie. Consultați licențele specifice seturilor de date (ex: ESC50) pentru restricții privind utilizarea datelor.
