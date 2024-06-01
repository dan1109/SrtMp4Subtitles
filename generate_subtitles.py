import sys
import os
import re
import time
import whisper
import numpy as np
import logging
import librosa
import soundfile as sf
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from dotenv import load_dotenv
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel, \
    QComboBox, QProgressBar, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from concurrent.futures import ThreadPoolExecutor

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
def write_segment_to_srt(file, idx, segment):
    start = convert_time(segment["start"])
    end = convert_time(segment["end"])
    text = segment["text"].strip()
    entry = f"{idx}\n{start} --> {end}\n{text}\n\n"
    file.write(entry)
    file.flush()  # Assicurati che i dati vengano scritti immediatamente


def find_silence_boundaries(audio, sample_rate, chunk_size=25, silence_threshold=0.02):
    non_silent = np.where(librosa.effects.split(audio, top_db=30))[0]
    boundaries = [0]
    last_boundary = 0
    for idx in range(0, len(non_silent), int(sample_rate * chunk_size)):
        boundary = non_silent[idx]
        if boundary - last_boundary > sample_rate * chunk_size:
            boundaries.append(boundary)
            last_boundary = boundary
    boundaries.append(len(audio))
    return boundaries


class SubtitleWorker(QThread):
    update_progress = pyqtSignal(int)
    log_message = pyqtSignal(str)

    def __init__(self, file_path, model_type):
        super().__init__()
        self.file_path = file_path
        self.model_type = model_type

    def run(self):
        # Percorso del file SRT
        srt_path = self.file_path.rsplit(".", 1)[0] + ".srt"

        # Carica l'audio e ottieni la frequenza di campionamento
        self.log_message.emit("Loading audio...")
        try:
            audio, sample_rate = librosa.load(self.file_path, sr=None)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)  # Converti in mono se l'audio ha più canali
        except Exception as e:
            self.log_message.emit(f"Error loading audio: {e}")
            raise

        self.log_message.emit(f"Audio loaded with sample rate: {sample_rate} Hz")

        # Trova i boundary dei segmenti utilizzando i silenzi
        self.log_message.emit("Finding silence boundaries...")
        boundaries = find_silence_boundaries(audio, sample_rate)
        self.log_message.emit(f"Found {len(boundaries) - 1} segments.")

        segments = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            segments.append((audio[start:end], start / sample_rate, end / sample_rate))

        # Carica il modello Whisper
        self.log_message.emit("Loading Whisper model...")
        model = whisper.load_model(self.model_type)
        self.log_message.emit("Whisper model loaded")

        # Processa i segmenti in parallelo
        def process_segment(segment):
            audio_segment, start_time, end_time = segment
            chunk_audio_path = "/tmp/chunk.wav"
            sf.write(chunk_audio_path, audio_segment, sample_rate)
            result = model.transcribe(chunk_audio_path, fp16=False, language="it", verbose=True)
            return result["segments"], start_time, end_time

        self.log_message.emit("Transcribing segments...")
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_segment, segments), total=len(segments), desc="Transcribing"))

        # Scrivi i segmenti nel file SRT
        self.log_message.emit("Writing segments to SRT file...")
        with open(srt_path, "w", encoding='utf-8') as srt_file:
            idx = 1
            for segment_result, start_time, end_time in results:
                for segment in segment_result:
                    segment["start"] += start_time
                    segment["end"] += start_time
                    write_segment_to_srt(srt_file, idx, segment)
                    idx += 1

        self.update_progress.emit(100)
        self.log_message.emit("Subtitles generated successfully.")


class SubtitleGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Subtitle SRT Generator')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        # Model selection combo box
        self.model_label = QLabel("Select Model Whisper Type:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["large", "medium", "small", "tiny"])
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)

        # File selection button
        self.file_button = QPushButton('Select Audio/Video File')
        self.file_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.file_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Log display
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Generate button
        self.generate_button = QPushButton('Generate Subtitles')
        self.generate_button.clicked.connect(self.generate_subtitles)
        layout.addWidget(self.generate_button)

        # Set the layout
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

    def clean_srt_file(self, srt_path):
        self.log_text.append(f"Cleaning SRT file: {srt_path}")

        if not os.path.exists(srt_path):
            self.log_text.append(f"File {srt_path} does not exist.")
            return

        with open(srt_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        cleaned_lines = []
        for line in lines:
            if self.is_valid_srt_line(line):
                cleaned_lines.append(line)

        with open(srt_path, 'w', encoding='utf-8') as file:
            file.writelines(cleaned_lines)

        self.log_text.append("SRT file cleaned successfully.")

    def is_valid_srt_line(self, line):
        # Placeholder for checking if a line in SRT is valid
        # Example: Remove lines with certain patterns or errors
        return True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SubtitleGeneratorApp()
    ex.show()
    sys.exit(app.exec_())
