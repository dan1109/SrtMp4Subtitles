
# generate_subtitles.py

## Descrizione
`generate_subtitles.py` è uno script Python che genera sottotitoli (.srt) a partire da file audio utilizzando il modello di trascrizione `whisper`. Il processo include la gestione dei chunk audio e l'aggiornamento continuo del file dei sottotitoli in tempo reale.

## Funzionalità
- Caricamento dei file audio.
- Divisione dell'audio in chunk gestibili.
- Trascrizione dei chunk audio in testo.
- Scrittura dei segmenti trascritti in un file .srt in tempo reale.
- Ripresa della trascrizione dal punto in cui era stata interrotta.

## Requisiti
- Python 3.x
- Librerie Python: `whisper`, `tqdm`, `numpy`, `librosa`, `soundfile`, `dotenv`, `PyQt5`
- File `.env` per le variabili d'ambiente

## Installazione
1. Clona questo repository:
    ```bash
    git clone https://github.com/your-username/generate_subtitles.git
    ```
2. Naviga nella directory del progetto:
    ```bash
    cd generate_subtitles
    ```
3. Installa le dipendenze:
    ```bash
    pip install whisper tqdm numpy librosa soundfile python-dotenv PyQt5
    ```

## Utilizzo
1. Esegui lo script:
    ```bash
    python generate_subtitles.py
    ```
2. Verrà aperta una finestra per selezionare il file audio da trascrivere.

## Esempio di utilizzo
```bash
python generate_subtitles.py
```

## Multilingua
### English

## Description
`generate_subtitles.py` is a Python script that generates subtitles (.srt) from audio files using the `whisper` transcription model. The process includes handling audio chunks and continuously updating the subtitles file in real-time.

## Features
- Load audio files.
- Split audio into manageable chunks.
- Transcribe audio chunks into text.
- Write transcribed segments to an .srt file in real-time.
- Resume transcription from where it was previously stopped.

## Requirements
- Python 3.x
- Python libraries: `whisper`, `tqdm`, `numpy`, `librosa`, `soundfile`, `dotenv`, `PyQt5`
- `.env` file for environment variables

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/generate_subtitles.git
    ```
2. Navigate to the project directory:
    ```bash
    cd generate_subtitles
    ```
3. Install the dependencies:
    ```bash
    pip install whisper tqdm numpy librosa soundfile python-dotenv PyQt5
    ```

## Usage
1. Run the script:
    ```bash
    python generate_subtitles.py
    ```
2. A window will open to select the audio file to be transcribed.

## Usage Example
```bash
python generate_subtitles.py
```

## Citazione
Se utilizzi questo script nel tuo progetto, per favore cita come segue:

```
@software{username_generate_subtitles_2024,
  author = {Daniele Bufalo},
  title = {generate_subtitles.py: Generazione di sottotitoli da file audio},
  year = {2024},
  url = {https://github.com/dan1109/SrtMp4Subtitles},
}
```

---

```
@software{username_generate_subtitles_2024,
  author = {Daniele Bufalo},
  title = {generate_subtitles.py: Subtitle generation from audio files},
  year = {2024},
  url = {https://github.com/dan1109/SrtMp4Subtitles},
}
```
