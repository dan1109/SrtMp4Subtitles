import whisper
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import logging
import librosa
import soundfile as sf

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Funzione per convertire il tempo in formato .srt
def convert_time(seconds):
    hr = int(seconds // 3600)
    min = int((seconds % 3600) // 60)
    sec = int(seconds % 60)
    ms = int((seconds * 1000) % 1000)
    return f"{hr:02}:{min:02}:{sec:02},{ms:03}"


# Funzione per convertire il tempo dal formato SRT a secondi
def srt_time_to_seconds(time_str):
    h, m, s = map(float, time_str.replace(',', '.').split(':'))
    return h * 3600 + m * 60 + s


# Funzione per scrivere i segmenti in un file .srt in tempo reale
def write_segment_to_srt(file, idx, segment, time_offset):
    start = convert_time(segment["start"] + time_offset)
    end = convert_time(segment["end"] + time_offset)
    text = segment["text"].strip()
    entry = f"{idx}\n{start} --> {end}\n{text}\n\n"
    file.write(entry)
    file.flush()  # Assicurati che i dati vengano scritti immediatamente


# Funzione per leggere l'ultimo indice di segmento salvato e il tempo finale
def read_last_index_and_time(srt_path):
    last_index = 0
    last_end_time = 0.0
    try:
        with open(srt_path, "r") as srt_file:
            lines = srt_file.readlines()
            if lines:
                for i in range(len(lines) - 1, -1, -1):
                    if "-->" in lines[i]:
                        end_time_str = lines[i].split(" --> ")[1].strip()
                        last_end_time = srt_time_to_seconds(end_time_str)
                        break
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().isdigit():
                        last_index = int(lines[i].strip())
                        break
    except Exception as e:
        logger.error(f"Errore durante la lettura del file SRT: {e}")
    return last_index, last_end_time


# Funzione per suddividere l'audio in chunk
def chunk_audio(audio, sample_rate, chunk_size=30):
    num_chunks = int(np.ceil(len(audio) / (chunk_size * sample_rate)))
    chunks = [audio[i * chunk_size * sample_rate:(i + 1) * chunk_size * sample_rate] for i in range(num_chunks)]
    return chunks


# Apri una finestra per selezionare il file video
Tk().withdraw()  # Nasconde la finestra principale di Tkinter
video_path = askopenfilename(title="Seleziona il file video", filetypes=[("MP4 files", "*.mp4")])

if not video_path:
    raise FileNotFoundError("Nessun file selezionato")

# Carica il modello Whisper
logger.info("Caricamento del modello Whisper...")
model = whisper.load_model("large")
logger.info("Modello Whisper caricato")

# Inizia il timer per la trascrizione
start_transcription = time.time()

# Percorso del file SRT
srt_path = video_path.rsplit(".", 1)[0] + ".srt"

# Ottieni l'ultimo indice e tempo finale salvato
last_index, last_end_time = read_last_index_and_time(srt_path)

# Carica l'audio e ottieni la frequenza di campionamento
logger.info("Caricamento dell'audio...")
try:
    audio, sample_rate = librosa.load(video_path, sr=None)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)  # Converti in mono se l'audio ha più canali
except Exception as e:
    logger.error(f"Errore durante il caricamento dell'audio: {e}")
    raise

logger.info(f"Audio caricato con frequenza di campionamento: {sample_rate} Hz")

# Calcola l'offset in campioni dall'ultimo tempo finale salvato
offset_samples = int(last_end_time * sample_rate)
audio = audio[offset_samples:]

# Suddividi l'audio in chunk
chunks = chunk_audio(audio, sample_rate)
logger.info(f"Audio suddiviso in {len(chunks)} chunk")

# Apri il file .srt per la scrittura in modalità append
with open(srt_path, "a") as srt_file:
    idx = last_index
    cumulative_time = last_end_time  # Inizializza il tempo cumulativo all'ultimo tempo finale salvato
    for chunk in tqdm(chunks, desc="Trascrizione in corso"):
        try:
            # Scrivi il chunk in un file temporaneo
            chunk_audio_path = "/tmp/chunk.wav"
            sf.write(chunk_audio_path, chunk, sample_rate)

            # Trascrivi l'audio raw
            result = model.transcribe(chunk_audio_path, fp16=False, language="it", verbose=True)
            segments = result["segments"]

            for segment in segments:
                idx += 1
                segment["start"] += cumulative_time
                segment["end"] += cumulative_time
                write_segment_to_srt(srt_file, idx, segment, cumulative_time)
                logger.info(f"Segmento {idx} trascritto e salvato")

            # Aggiorna il tempo cumulativo
            cumulative_time += len(chunk) / sample_rate

        except Exception as e:
            logger.error(f"Errore durante la trascrizione del chunk: {e}")

# Chiudi il file .srt alla fine per assicurarsi che tutto venga salvato
srt_file.close()

# Fine del timer per la trascrizione
end_transcription = time.time()
transcription_time = end_transcription - start_transcription

# Output del tempo impiegato
logger.info(f"Tempo di trascrizione: {transcription_time // 60:.0f} minuti e {transcription_time % 60:.0f} secondi")
logger.info(f"Sottotitoli salvati in: {srt_path}")
