import os
import concurrent.futures
import multiprocessing
import sys
import time
import psutil
import logging
from datetime import datetime
import numpy as np
import librosa
from tqdm import tqdm
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QWidget, QFileDialog, QLabel, QComboBox, QProgressBar, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from concurrent.futures import ProcessPoolExecutor
import whisper
from dotenv import load_dotenv

# Carica variabili d'ambiente e configura logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_AVAILABLE_MEMORY = 500 * 1024 * 1024  # 500 MB

# --------------------- FUNZIONI DI SUPPORTO --------------------- #

def estimate_model_memory_usage(model_type, attempts=3):
    import gc
    usage_values = []
    for _ in range(attempts):
        gc.collect()
        mem_before = psutil.virtual_memory().available
        temp_model = whisper.load_model(model_type)
        gc.collect()
        mem_after = psutil.virtual_memory().available

        used_mb = (mem_before - mem_after) / (1024 * 1024)
        usage_values.append(max(used_mb, 0))

        del temp_model
        gc.collect()

    return sum(usage_values) / len(usage_values)


def safe_share_memory(model):
    for param in model.parameters():
        if not param.is_sparse:
            try:
                param.share_memory_()
            except Exception as e:
                logger.warning(f"Impossibile condividere parametro: {e}")
    for buf in model.buffers():
        if not buf.is_sparse:
            try:
                buf.share_memory_()
            except Exception as e:
                logger.warning(f"Impossibile condividere buffer: {e}")
    return model

BASELINE_MEMORY_MB = 300  # stima dell'overhead di un processo vuoto (da regolare in base al sistema)

def worker(q, model_type):
    import whisper
    import psutil, os, time
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss  # memoria prima del caricamento
    model = whisper.load_model(model_type, device='cpu')
    time.sleep(0.5)  # attesa per stabilizzare la misurazione
    after = process.memory_info().rss  # memoria dopo il caricamento
    usage = after - baseline
    q.put(usage)
    time.sleep(1)  # mantieni attivo il worker per la misurazione

def test_worker_memory_usage(model_type):
    import multiprocessing
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(q, model_type))
    p.start()
    usage_bytes = q.get()  # incremento in byte
    p.join()
    usage_mb = usage_bytes / (1024 * 1024)
    # Sottrai la baseline stimata per isolare il consumo aggiuntivo del modello
    adjusted_usage = usage_mb - BASELINE_MEMORY_MB
    logger.info(f"Uso stimato per worker (grezzo): {usage_mb:.2f} MB, " +
                f"adjusted (baseline sottratta): {adjusted_usage:.2f} MB")
    return max(adjusted_usage, 0)


def calculate_safe_worker_count(model_type):
    import gc
    safety_margin = 1 * 1024 * 1024 * 1024  # 1GB di margine
    initial_available = psutil.virtual_memory().available
    logger.info(f"Memoria disponibile iniziale: {initial_available / (1024 * 1024):.2f} MB")

    if initial_available < safety_margin:
        raise MemoryError("Memoria insufficiente per avviare qualsiasi worker")

    # Esegui più simulazioni e calcola la media del consumo incrementale corretto
    usages = []
    for _ in range(3):
        usage = test_worker_memory_usage(model_type)
        usages.append(usage)
    avg_usage = sum(usages) / len(usages)
    logger.info(f"Media memoria usata (adjusted) da un worker testato: {avg_usage:.2f} MB")

    # Calcola quanti worker possono essere avviati in sicurezza
    max_workers_by_memory = int((initial_available - safety_margin) / (avg_usage * 1024 * 1024))
    cpu_workers = max(1, os.cpu_count() - 2)
    safe_workers = int(min(max_workers_by_memory, cpu_workers, 10))

    # Forza un minimo di 1 (o 2 se possibile e la memoria residua lo permette)
    if safe_workers < 2 and (initial_available - safety_margin) > (avg_usage * 2 * 1024 * 1024):
        safe_workers = 2

    safe_workers = max(1, safe_workers)
    logger.info(f"Safe worker count (calcolato): {safe_workers}")
    return safe_workers


def convert_time(seconds):
    hr = int(seconds // 3600)
    min_ = int((seconds % 3600) // 60)
    sec = int(seconds % 60)
    ms = int((seconds * 1000) % 1000)
    return f"{hr:02}:{min_:02}:{sec:02},{ms:03}"


def process_segment(args):
    audio_chunk, chunk_start, chunk_end, sample_rate = args
    logger.info(f"Inizio chunk: {chunk_start:.2f}s -> {chunk_end:.2f}s")

    audio_chunk = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=16000)
    audio_chunk = audio_chunk.astype(np.float32)

    result = model.transcribe(
        audio_chunk,
        language="it",
        fp16=worker_fp16,
        temperature=0.0,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        verbose=False
    )

    logger.info(f"Fine chunk: {chunk_start:.2f}s -> {chunk_end:.2f}s")
    return result["segments"], chunk_start, chunk_end


def write_segment_to_srt(file, idx, segment):
    start = convert_time(segment["start"])
    end = convert_time(segment["end"])
    text = segment["text"].strip()
    entry = f"{idx}\n{start} --> {end}\n{text}\n\n"
    file.write(entry)
    file.flush()


def split_audio_by_silence(audio, sample_rate, top_db=80, min_chunk_duration=5):
    intervals = librosa.effects.split(audio, top_db=top_db)
    chunks = []
    for start, end in intervals:
        duration = (end - start) / sample_rate
        if duration < min_chunk_duration:
            continue
        chunks.append((audio[start:end], start / sample_rate, end / sample_rate))
    # Se non si ottengono almeno due chunk, suddividi uniformemente l'audio
    if len(chunks) < 2 and len(audio) / sample_rate > min_chunk_duration:
        num_segments = int(len(audio) / (sample_rate * min_chunk_duration))
        segment_length = int(sample_rate * min_chunk_duration)
        chunks = []
        for i in range(num_segments):
            start = i * segment_length
            end = min((i + 1) * segment_length, len(audio))
            chunks.append((audio[start:end], start / sample_rate, end / sample_rate))
    return chunks

# --- Funzioni per evitare duplicazioni di segmenti già elaborati ---

def parse_time(time_str):
    """Converte una stringa nel formato 'HH:MM:SS,ms' in secondi."""
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def get_last_end_time(srt_path):
    """Recupera l'ultimo tempo di fine (in secondi) dal file SRT, se esiste."""
    if not os.path.exists(srt_path):
        return 0.0
    with open(srt_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for line in reversed(lines):
        if "-->" in line:
            parts = line.split("-->")
            if len(parts) == 2:
                end_str = parts[1].strip()
                return parse_time(end_str)
    return 0.0

def get_last_index(srt_path):
    """Recupera l'ultimo indice numerico presente nel file SRT, se esiste."""
    if not os.path.exists(srt_path):
        return 0
    last_index = 0
    with open(srt_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for line in reversed(lines):
        if line.strip().isdigit():
            last_index = int(line.strip())
            break
    return last_index

def init_worker(model_type):
    global model, worker_fp16
    device = 'cpu'
    worker_fp16 = False
    logger.info(f"Caricamento modello Whisper su {device} nel worker")
    model = whisper.load_model(model_type, device=device)


# --------------------- CLASSE PER LA TRASCRIZIONE --------------------- #

class SubtitleWorker(QThread):
    update_progress = pyqtSignal(int)
    log_message = pyqtSignal(str)

    def __init__(self, file_path, model_type):
        super().__init__()
        self.file_path = file_path
        self.model_type = model_type

    def run(self):
        self.log_message.emit("Inizio esecuzione thread di sottotitolazione.")
        srt_path = self.file_path.rsplit(".", 1)[0] + ".srt"

        self.log_message.emit("Loading audio...")
        try:
            audio, sample_rate = librosa.load(self.file_path, sr=None)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            self.log_message.emit(f"Audio caricato: {len(audio)} campioni a {sample_rate} Hz")
        except Exception as e:
            self.log_message.emit(f"Error loading audio: {e}")
            return

        try:
            safe_worker_count = calculate_safe_worker_count(self.model_type)
            self.log_message.emit(f"Safe worker count calcolato: {safe_worker_count}")
        except Exception as e:
            self.log_message.emit(f"Errore nel calcolo del safe worker count: {e}")
            return

        detected_chunks = split_audio_by_silence(audio, sample_rate)
        self.log_message.emit(f"Numero di chunk rilevati (basati sui silenzi): {len(detected_chunks)}")

        # --- Nuova logica per evitare elaborazioni duplicate ---
        if os.path.exists(srt_path):
            last_end = get_last_end_time(srt_path)
            self.log_message.emit(f"Ultimo tempo elaborato dal SRT: {last_end:.2f} s")
            new_chunks = []
            for audio_chunk, start, end in detected_chunks:
                if end <= last_end:
                    continue  # Chunk già elaborato
                elif start < last_end < end:
                    offset = int((last_end - start) * sample_rate)
                    if offset < len(audio_chunk):
                        new_audio_chunk = audio_chunk[offset:]
                        new_chunks.append((new_audio_chunk, last_end, end))
                else:
                    new_chunks.append((audio_chunk, start, end))
            detected_chunks = new_chunks
            self.log_message.emit(f"Numero di chunk da elaborare dopo filtraggio: {len(detected_chunks)}")
            if len(detected_chunks) == 0:
                self.log_message.emit("Tutti i segmenti sono già stati elaborati. Operazione terminata.")
                self.update_progress.emit(100)
                return
        # --- Fine logica duplicazioni ---

        chunks_prepared = [
            (librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=16000), start, end, 16000)
            for audio_chunk, start, end in detected_chunks
        ]

        start_time = datetime.now()
        self.log_message.emit(f"Avvio trascrizione con processi paralleli: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        total_chunks = len(chunks_prepared)
        completed = 0

        current_index = get_last_index(srt_path) + 1

        try:
            with ProcessPoolExecutor(
                    initializer=init_worker,
                    initargs=(self.model_type,),
                    max_workers=safe_worker_count
            ) as executor:
                future_to_chunk = {executor.submit(process_segment, chunk): chunk for chunk in chunks_prepared}
                for future in concurrent.futures.as_completed(future_to_chunk):
                    if future.exception():
                        self.log_message.emit(f"Errore worker: {future.exception()}")
                        continue
                    segments, chunk_start, chunk_end = future.result()
                    for seg in segments:
                        seg["start"] += chunk_start
                        seg["end"] += chunk_start
                    # Apertura in append per aggiornare il file SRT immediatamente per il chunk elaborato
                    with open(srt_path, "a", encoding="utf-8") as srt_file:
                        for seg in segments:
                            write_segment_to_srt(srt_file, current_index, seg)
                            current_index += 1
                    completed += 1
                    progress = int((completed / total_chunks) * 100)
                    self.update_progress.emit(progress)

        except Exception as e:
            self.log_message.emit(f"Errore durante la trascrizione: {e}")
            return

        end_time = datetime.now()
        self.log_message.emit(f"Fine: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        duration = end_time - start_time
        self.log_message.emit(f"Durata totale: {str(duration)}")
        self.update_progress.emit(100)
        self.log_message.emit("Sottotitoli generati con successo!")


# --------------------- INTERFACCIA GRAFICA --------------------- #

class SubtitleGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Subtitle SRT Generator")
        self.setGeometry(100, 100, 600, 400)
        layout = QVBoxLayout()

        self.model_label = QLabel("Select Model Whisper Type:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["large", "medium", "small", "tiny"])
        self.model_combo.setCurrentText("large")
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)

        self.file_button = QPushButton("Select Audio/Video File")
        self.file_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.file_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.generate_button = QPushButton("Generate Subtitles")
        self.generate_button.clicked.connect(self.generate_subtitles)
        layout.addWidget(self.generate_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_filter = "Audio/Video Files (*.mp4 *.avi *.mkv *.mp3 *.wav)"
        self.file_name, _ = QFileDialog.getOpenFileName(self, "Select Audio/Video File", "", file_filter,
                                                        options=options)
        if self.file_name:
            self.log_text.append(f"Selected file: {self.file_name}")

    def generate_subtitles(self):
        if not hasattr(self, 'file_name'):
            self.log_text.append("No file selected.")
            return
        self.log_text.append("Generating subtitles...")
        self.progress_bar.setValue(0)
        model_type = self.model_combo.currentText()
        self.worker = SubtitleWorker(self.file_name, model_type)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.log_message.connect(self.log_message)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def log_message(self, message):
        self.log_text.append(message)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    app = QApplication(sys.argv)
    ex = SubtitleGeneratorApp()
    ex.show()
    sys.exit(app.exec_())
