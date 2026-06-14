from __future__ import annotations

import gc
import json
import os
import re
import sys
import threading
import wave
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

import gradio as gr
from faster_whisper import WhisperModel

APP_TITLE = "Сервис транскрибации аудио"
APP_SUBTITLE = "Загрузите голосовое сообщение или аудиофайл, и получите текст"

SUPPORTED_EXTENSIONS = [
    ".mp3", ".wav", ".m4a", ".ogg", ".opus", ".aac", ".mp4", ".mpeg", ".webm"
]
SUPPORTED_EXTENSIONS_SET = set(SUPPORTED_EXTENSIONS)


def get_runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_output_root_dir() -> Path:
    if getattr(sys, "frozen", False):
        documents_dir = Path.home() / "Documents"
        base_dir = documents_dir if documents_dir.exists() else Path.home()
        return base_dir / "Transkribator"
    return Path.cwd()


RUNTIME_BASE_DIR = get_runtime_base_dir()
APP_CONFIG_DIR = Path.home() / ".transkribator"
APP_CONFIG_PATH = APP_CONFIG_DIR / "config.json"


def load_app_config() -> dict:
    try:
        if APP_CONFIG_PATH.exists():
            return json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_app_config(config: dict) -> None:
    APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    APP_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def choose_output_root_interactive(initial_dir: Path | None = None) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            title="Выберите папку для сохранения результатов транскрибации",
            initialdir=str(initial_dir or Path.home()),
            mustexist=True,
        )
        root.destroy()
        if selected:
            return Path(selected).expanduser().resolve()
    except Exception:
        return None
    return None


def initialize_output_root_dir() -> Path:
    default_root = get_output_root_dir()
    config = load_app_config()
    saved_path = config.get("output_root")

    if saved_path:
        try:
            output_root = Path(saved_path).expanduser().resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            return output_root
        except Exception:
            pass

    chosen_path = choose_output_root_interactive(default_root)
    output_root = chosen_path or default_root
    output_root.mkdir(parents=True, exist_ok=True)
    save_app_config({"output_root": str(output_root)})
    return output_root


OUTPUT_ROOT_DIR = initialize_output_root_dir()
OUTPUT_DIR = OUTPUT_ROOT_DIR / "transcription_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Профили с упором на качество речи, а не только на скорость
MODEL_PROFILES = {
    "Быстро": "base",
    "Баланс": "small",
    "Точнее": "medium",
}

LANGUAGE_MAP = {
    "Авто": None,
    "Русский": "ru",
    "Английский": "en",
}

RESULT_MODE_WITH_TIMESTAMPS = "Текст + таймкоды"
# Крупнее чанки = меньше накладных расходов на длинных файлах
CHUNK_SECONDS = 30 * 60
# Короткие файлы не режем вообще
CHUNKING_THRESHOLD_SECONDS = 35 * 60
CPU_THREADS = max(1, (os.cpu_count() or 4) - 2)

_MODEL_CACHE: dict[tuple[str, str], WhisperModel] = {}
_FFMPEG_BIN: str | None = None
_CANCEL_EVENT = threading.Event()


class FriendlyError(Exception):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class ProcessingCancelled(Exception):
    pass


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(name)).strip()
    return cleaned or "result"


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def format_seconds(seconds: float) -> str:
    value = max(0, int(round(seconds)))
    hours = value // 3600
    minutes = (value % 3600) // 60
    secs = value % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_output_path_label() -> str:
    return str(OUTPUT_DIR)


def normalize_text(text: str) -> str:
    text = (text or "").replace("\n", " ").replace("\t", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def improve_punctuation_style(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return text
    text = re.sub(r"([.!?])([^\s])", r"\1 \2", text)
    text = re.sub(r"([,:;])([^\s])", r"\1 \2", text)
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]

    def uppercase_after_punct(match):
        return match.group(1) + match.group(2).upper()

    text = re.sub(r"([.!?]\s+)([a-zа-я])", uppercase_after_punct, text, flags=re.IGNORECASE)
    return text.strip()


def validate_uploaded_files(file_paths) -> list[str]:
    if not file_paths:
        raise FriendlyError("Файл не загружен", "Загрузите хотя бы один аудио- или видеофайл.")

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    normalized = [str(item) for item in file_paths if item]
    if not normalized:
        raise FriendlyError("Файл не загружен", "Загрузите хотя бы один аудио- или видеофайл.")

    for file_path in normalized:
        path = Path(file_path)
        if not path.exists():
            raise FriendlyError("Ошибка загрузки", f"Файл «{path.name}» не найден.")
        if path.stat().st_size == 0:
            raise FriendlyError("Пустой файл", f"Файл «{path.name}» пустой.")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS_SET:
            raise FriendlyError(
                "Неподдерживаемый формат",
                "Поддерживаются: " + ", ".join(SUPPORTED_EXTENSIONS)
            )

    return normalized


def throw_if_cancelled() -> None:
    if _CANCEL_EVENT.is_set():
        raise ProcessingCancelled()


def find_ffmpeg() -> str:
    global _FFMPEG_BIN
    if _FFMPEG_BIN:
        return _FFMPEG_BIN

    candidates = [
        str(RUNTIME_BASE_DIR / "ffmpeg.exe"),
        str(RUNTIME_BASE_DIR / "ffmpeg"),
        shutil.which("ffmpeg"),
        "ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        r"C:\\ffmpeg\\bin\\ffmpeg.exe",
        r"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        r"C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe",
    ]

    checked: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.append(candidate)
        try:
            result = subprocess.run(
                [candidate, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0:
                _FFMPEG_BIN = candidate
                return candidate
        except FileNotFoundError:
            continue

    raise FriendlyError(
        "ffmpeg не найден",
        "Не найден ffmpeg. Установите его и добавьте в PATH. Для macOS: brew install ffmpeg. Для Windows: установите ffmpeg и добавьте путь к ffmpeg.exe в PATH."
    )


def convert_to_wav(input_path: str, output_wav_path: Path) -> None:
    throw_if_cancelled()
    ffmpeg_bin = find_ffmpeg()

    command = [
        ffmpeg_bin,
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_wav_path),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0 or not output_wav_path.exists():
        raise FriendlyError("Ошибка обработки аудио", "Не удалось подготовить файл к распознаванию.")

    if output_wav_path.stat().st_size == 0:
        raise FriendlyError("Ошибка обработки аудио", "После конвертации аудио оказалось пустым.")


def get_audio_duration_from_wav(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if frame_rate <= 0:
                return 0.0
            return frame_count / float(frame_rate)
    except Exception:
        raise FriendlyError("Ошибка чтения аудио", "Не удалось прочитать подготовленное аудио.")


def split_wav_into_chunks(wav_path: Path, temp_dir: Path, chunk_seconds: int = CHUNK_SECONDS):
    items = []
    with wave.open(str(wav_path), "rb") as wav_file:
        params = wav_file.getparams()
        frame_rate = wav_file.getframerate()
        total_frames = wav_file.getnframes()

        if frame_rate <= 0 or total_frames <= 0:
            raise FriendlyError("Пустой файл", "Файл пустой или не содержит аудио.")

        total_duration = total_frames / float(frame_rate)
        if total_duration < 0.4:
            raise FriendlyError("Слишком короткое аудио", "Аудио слишком короткое для распознавания.")

        frames_per_chunk = max(int(chunk_seconds * frame_rate), frame_rate)
        current_start_frame = 0
        chunk_index = 1

        while current_start_frame < total_frames:
            frames_to_read = min(frames_per_chunk, total_frames - current_start_frame)
            raw_frames = wav_file.readframes(frames_to_read)

            chunk_path = temp_dir / f"chunk_{chunk_index:03d}.wav"
            with wave.open(str(chunk_path), "wb") as chunk_wav:
                chunk_wav.setparams(params)
                chunk_wav.writeframes(raw_frames)

            items.append({
                "path": str(chunk_path),
                "offset": current_start_frame / float(frame_rate),
            })
            current_start_frame += frames_to_read
            chunk_index += 1

    return items, total_duration


def get_model(model_size: str) -> WhisperModel:
    key = ("cpu", model_size)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=CPU_THREADS,
            num_workers=1,
        )
    return _MODEL_CACHE[key]


def build_readable_text(segments: list[dict], improve: bool = False) -> str:
    if not segments:
        return ""

    paragraphs = []
    current_parts = []
    current_length = 0
    previous_end = None

    for segment in segments:
        seg_text = normalize_text(segment["text"])
        if not seg_text:
            continue

        pause = 0 if previous_end is None else max(0, segment["start"] - previous_end)
        if current_parts and (pause >= 2.2 or current_length >= 450):
            paragraph = " ".join(current_parts).strip()
            if paragraph:
                paragraphs.append(paragraph)
            current_parts = []
            current_length = 0

        current_parts.append(seg_text)
        current_length += len(seg_text)
        previous_end = segment["end"]

    if current_parts:
        paragraph = " ".join(current_parts).strip()
        if paragraph:
            paragraphs.append(paragraph)

    result = "\n\n".join(paragraphs).strip()
    return improve_punctuation_style(result) if improve else result


def build_timestamps_text(segments: list[dict], improve: bool = False) -> str:
    lines = []
    for segment in segments:
        seg_text = normalize_text(segment["text"])
        if not seg_text:
            continue
        if improve:
            seg_text = improve_punctuation_style(seg_text)
        lines.append(f"[{format_seconds(segment['start'])} - {format_seconds(segment['end'])}] {seg_text}")
    return "\n".join(lines).strip()


def transcribe_chunk(model: WhisperModel, chunk_path: str, language: str | None, model_size: str):
    throw_if_cancelled()
    beam_size = 3 if model_size == "base" else 5
    best_of = 3 if model_size == "base" else 5
    try:
        segments_iter, info = model.transcribe(
            chunk_path,
            language=language,
            vad_filter=False,
            word_timestamps=False,
            beam_size=beam_size,
            best_of=best_of,
            condition_on_previous_text=True,
            temperature=0.0,
        )
        return list(segments_iter), info
    except Exception:
        raise FriendlyError("Ошибка модели", "Не удалось распознать аудиофайл.")


def transcribe_single_file(
    file_path: str,
    model_size: str,
    language: str | None,
    with_timestamps: bool,
    improve_punctuation: bool,
    progress,
    file_label: str,
) -> tuple[str, str, str]:
    temp_dir = Path(tempfile.mkdtemp(prefix="fw_transcription_"))
    try:
        throw_if_cancelled()
        wav_path = temp_dir / "prepared.wav"
        progress(0.06, desc=f"{file_label}: подготавливаем аудио")
        convert_to_wav(file_path, wav_path)

        duration = get_audio_duration_from_wav(wav_path)
        if duration < 0.4:
            raise FriendlyError("Слишком короткое аудио", f"Файл «{Path(file_path).name}» слишком короткий.")

        # Для коротких и средних файлов не режем аудио, это быстрее
        if duration <= CHUNKING_THRESHOLD_SECONDS:
            chunk_items = [{"path": str(wav_path), "offset": 0.0}]
        else:
            chunk_items, _ = split_wav_into_chunks(wav_path, temp_dir, CHUNK_SECONDS)

        model = get_model(model_size)

        all_segments = []
        detected_language = language
        total_chunks = len(chunk_items)

        for index, chunk in enumerate(chunk_items, start=1):
            throw_if_cancelled()
            fraction = 0.12 + 0.80 * ((index - 1) / max(total_chunks, 1))
            if total_chunks == 1:
                progress(fraction, desc=f"{file_label}: распознаем аудио")
            else:
                progress(fraction, desc=f"{file_label}: распознаем часть {index} из {total_chunks}")

            segments, info = transcribe_chunk(model, chunk["path"], detected_language, model_size)

            if detected_language is None and getattr(info, "language", None):
                detected_language = info.language

            for segment in segments:
                seg_text = normalize_text(getattr(segment, "text", ""))
                if not seg_text:
                    continue
                start_time = float(getattr(segment, "start", 0.0)) + chunk["offset"]
                end_time = float(getattr(segment, "end", 0.0)) + chunk["offset"]
                all_segments.append({
                    "start": start_time,
                    "end": max(start_time, end_time),
                    "text": seg_text,
                })

        if not all_segments:
            raise FriendlyError("Нет распознанной речи", f"В файле «{Path(file_path).name}» не удалось распознать речь.")

        full_text = build_readable_text(all_segments, improve=improve_punctuation)
        timestamps_text = build_timestamps_text(all_segments, improve=improve_punctuation) if with_timestamps else ""

        progress(0.97, desc=f"{file_label}: сохраняем результат")
        return full_text, timestamps_text, detected_language or "не определён"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def save_results_for_file(source_path: str, full_text: str, timestamps_text: str, include_timestamps: bool) -> list[str]:
    source = Path(source_path)
    stem = sanitize_filename(source.stem)
    created = []

    text_path = unique_path(OUTPUT_DIR, f"{stem}_transcript", ".txt")
    text_path.write_text(full_text, encoding="utf-8")
    created.append(str(text_path))

    if include_timestamps and timestamps_text.strip():
        timestamps_path = unique_path(OUTPUT_DIR, f"{stem}_timestamps", ".txt")
        timestamps_path.write_text(timestamps_text, encoding="utf-8")
        created.append(str(timestamps_path))

    return created


def build_combined_text(results: list[dict]) -> str:
    return "\n\n".join(f"===== {item['file_name']} =====\n{item['text']}" for item in results).strip()


def build_combined_timestamps(results: list[dict]) -> str:
    return "\n\n".join(
        f"===== {item['file_name']} =====\n{item['timestamps']}"
        for item in results
        if item.get("timestamps", "").strip()
    ).strip()


def save_combined_files(results: list[dict], include_timestamps: bool) -> tuple[str, str | None]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_text_path = unique_path(OUTPUT_DIR, f"all_transcripts_{timestamp}", ".txt")
    combined_text_path.write_text(build_combined_text(results), encoding="utf-8")

    combined_timestamps_path = None
    if include_timestamps:
        content = build_combined_timestamps(results)
        if content.strip():
            path = unique_path(OUTPUT_DIR, f"all_timestamps_{timestamp}", ".txt")
            path.write_text(content, encoding="utf-8")
            combined_timestamps_path = str(path)

    return str(combined_text_path), combined_timestamps_path


def create_zip_archive(file_paths: list[str]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = unique_path(OUTPUT_DIR, f"transcriptions_batch_{timestamp}", ".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists():
                archive.write(path, arcname=path.name)
    return str(zip_path)


def clear_ui():
    _CANCEL_EVENT.clear()
    return (
        "Загрузите один или несколько файлов и нажмите «Транскрибировать».",
        "",
        gr.update(value="", visible=False),
        [],
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
    )


def change_output_directory():
    global OUTPUT_ROOT_DIR, OUTPUT_DIR

    chosen_path = choose_output_root_interactive(OUTPUT_ROOT_DIR)
    if not chosen_path:
        return (
            "Папка сохранения не изменена.",
            gr.update(value=get_output_path_label()),
        )

    OUTPUT_ROOT_DIR = chosen_path
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = OUTPUT_ROOT_DIR / "transcription_outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_app_config({"output_root": str(OUTPUT_ROOT_DIR)})

    return (
        f"Папка сохранения обновлена: {OUTPUT_DIR}",
        gr.update(value=get_output_path_label()),
    )


def request_cancel():
    _CANCEL_EVENT.set()
    return "Остановка запрошена. Сервис сохранит уже готовые результаты и завершит обработку на ближайшем безопасном этапе."


def process_files(
    file_paths,
    language_choice,
    result_mode,
    improve_punctuation,
    speed_profile,
    progress=gr.Progress(track_tqdm=False),
):
    empty_ts = gr.update(value="", visible=False)
    empty_files = gr.update(value=None, visible=False)
    empty_btn = gr.update(value=None, visible=False)

    try:
        _CANCEL_EVENT.clear()
        validated = validate_uploaded_files(file_paths)
        with_timestamps = result_mode == RESULT_MODE_WITH_TIMESTAMPS
        model_size = MODEL_PROFILES[speed_profile]
        language = LANGUAGE_MAP[language_choice]

        # Греем модель один раз заранее, чтобы первый файл не казался зависшим
        progress(0.01, desc="Подготавливаем модель")
        get_model(model_size)

        success_results = []
        file_rows = []
        created_files = []
        total_files = len(validated)

        for index, file_path in enumerate(validated, start=1):
            file_name = Path(file_path).name

            def file_progress(local_fraction, desc):
                overall = ((index - 1) + local_fraction) / total_files
                progress(overall, desc=desc)

            try:
                throw_if_cancelled()
                full_text, timestamps_text, detected_language = transcribe_single_file(
                    file_path=file_path,
                    model_size=model_size,
                    language=language,
                    with_timestamps=with_timestamps,
                    improve_punctuation=improve_punctuation,
                    progress=file_progress,
                    file_label=file_name,
                )

                created = save_results_for_file(
                    source_path=file_path,
                    full_text=full_text,
                    timestamps_text=timestamps_text,
                    include_timestamps=with_timestamps,
                )
                created_files.extend(created)

                success_results.append({
                    "file_name": file_name,
                    "text": full_text,
                    "timestamps": timestamps_text,
                })

                file_rows.append([file_name, "Готово", "", detected_language, "Обработка завершена"])

            except ProcessingCancelled:
                file_rows.append([file_name, "Остановлено", "", "", "Обработка остановлена пользователем"])
                break
            except FriendlyError as e:
                file_rows.append([file_name, "Ошибка", e.error_type, "", e.message])
            except Exception as e:
                file_rows.append([file_name, "Ошибка", type(e).__name__, "", str(e) or "Неожиданная ошибка"])

            gc.collect()

        errors = [row for row in file_rows if row[1] == "Ошибка"]
        was_cancelled = _CANCEL_EVENT.is_set()
        error_summary = "\n".join(f"- {row[0]} | {row[2]} | {row[4]}" for row in errors)

        if not success_results:
            if was_cancelled:
                return (
                    "Обработка остановлена. До остановки не успел завершиться ни один файл.",
                    "",
                    empty_ts,
                    file_rows,
                    empty_files,
                    empty_btn,
                    gr.update(value="Остановка выполнена по запросу пользователя.", visible=True),
                )
            return (
                "Ошибка обработки: не удалось обработать ни один файл.",
                "",
                empty_ts,
                file_rows,
                empty_files,
                empty_btn,
                gr.update(value=error_summary, visible=True),
            )

        combined_text_path, combined_timestamps_path = save_combined_files(
            success_results,
            include_timestamps=with_timestamps,
        )
        created_files.append(combined_text_path)
        if combined_timestamps_path:
            created_files.append(combined_timestamps_path)

        zip_path = create_zip_archive(created_files)
        combined_text = build_combined_text(success_results)
        combined_timestamps = build_combined_timestamps(success_results)

        if was_cancelled:
            status = f"Обработка остановлена. Успешно сохранено файлов: {len(success_results)}."
        elif errors:
            status = f"Обработка завершена. Успешно: {len(success_results)}. С ошибками: {len(errors)}."
        else:
            status = f"Готово. Успешно обработано файлов: {len(success_results)}."

        progress(1.0, desc="Готово")

        summary_lines = []
        if error_summary:
            summary_lines.append(error_summary)
        if was_cancelled:
            summary_lines.append("Обработка остановлена по запросу пользователя. Частичный прогресс сохранён.")

        return (
            status,
            combined_text,
            gr.update(value=combined_timestamps, visible=bool(with_timestamps and combined_timestamps.strip())),
            file_rows,
            gr.update(value=created_files, visible=True),
            gr.update(value=zip_path, visible=True),
            gr.update(value="\n".join(summary_lines), visible=bool(summary_lines)),
        )

    except FriendlyError as e:
        return (
            e.message,
            "",
            empty_ts,
            [],
            empty_files,
            empty_btn,
            gr.update(value=f"{e.error_type}: {e.message}", visible=True),
        )
    except Exception as e:
        return (
            f"Непредвиденная ошибка: {type(e).__name__}",
            "",
            empty_ts,
            [],
            empty_files,
            empty_btn,
            gr.update(value=f"{type(e).__name__}: {str(e)}", visible=True),
        )


custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding-top: 10px !important;
}
.hero {
    padding: 18px;
    border-radius: 18px;
    background: linear-gradient(180deg, #ffffff 0%, #f6fbff 100%);
    border: 1px solid #dbe7f3;
}
.hero-title {
    font-size: 30px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 6px;
}
.hero-subtitle {
    font-size: 15px;
    line-height: 1.6;
    color: #475569;
}
.note-box {
    background: #fbfdff;
    border: 1px solid #e4edf6;
    border-radius: 16px;
    padding: 12px 14px;
    font-size: 14px;
    line-height: 1.6;
    color: #334155;
}
#result_box textarea,
#error_box textarea,
#timestamps_box textarea {
    font-size: 15px !important;
    line-height: 1.68 !important;
}
footer {
    visibility: hidden;
}
"""

with gr.Blocks() as demo:
    gr.HTML(f"""
        <div class="hero">
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">{APP_SUBTITLE}</div>
        </div>
    """)

    gr.HTML(f"""
        <div class="note-box">
            Поддерживаются: {", ".join(SUPPORTED_EXTENSIONS)}<br>
            При первом запуске приложение попросит выбрать папку для сохранения результатов.<br>
            Короткие файлы сервис обрабатывает без лишнего разбиения, длинные — по частям.
        </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            files_input = gr.File(
                label="Загрузить один или несколько файлов",
                file_types=["audio", "video", ".opus", ".ogg", ".m4a", ".aac", ".mpeg", ".webm"],
                file_count="multiple",
                type="filepath",
                height=180,
            )

        with gr.Column(scale=4):
            language_dropdown = gr.Dropdown(
                choices=["Авто", "Русский", "Английский"],
                value="Авто",
                label="Язык",
            )
            result_mode = gr.Radio(
                choices=["Только текст", "Текст + таймкоды"],
                value="Только текст",
                label="Что показать в результате",
            )
            speed_profile = gr.Radio(
                choices=["Быстро", "Баланс", "Точнее"],
                value="Быстро",
                label="Профиль скорости",
            )
            punctuation_checkbox = gr.Checkbox(
                value=True,
                label="Улучшить пунктуацию",
            )

    status_box = gr.Textbox(
        label="Статус",
        value="Загрузите один или несколько файлов и нажмите «Транскрибировать».",
        interactive=False,
    )

    output_path_box = gr.Textbox(
        label="Папка сохранения результатов",
        value=get_output_path_label(),
        interactive=False,
    )

    with gr.Row():
        transcribe_button = gr.Button("Транскрибировать", variant="primary")
        stop_button = gr.Button("Остановить", variant="stop")
        change_output_button = gr.Button("Изменить папку сохранения")
        clear_button = gr.Button("Очистить", variant="secondary")

    with gr.Tabs():
        with gr.Tab("Результат"):
            result_text = gr.Textbox(
                label="Результат",
                lines=18,
                max_lines=30,
                placeholder="Здесь появится распознанный текст",
                elem_id="result_box",
            )
            timestamps_text = gr.Textbox(
                label="Таймкоды",
                lines=12,
                max_lines=20,
                placeholder="Здесь появятся таймкоды",
                visible=False,
                elem_id="timestamps_box",
            )

        with gr.Tab("Статус по файлам"):
            result_table = gr.Dataframe(
                headers=["Файл", "Статус", "Тип ошибки", "Язык", "Комментарий"],
                datatype=["str", "str", "str", "str", "str"],
                row_count=1,
                column_count=(5, "fixed"),
                interactive=False,
                wrap=True,
            )
            error_box = gr.Textbox(
                label="Ошибки",
                visible=False,
                lines=8,
                elem_id="error_box",
            )

        with gr.Tab("Скачать"):
            result_files = gr.File(
                label="Готовые файлы",
                file_count="multiple",
                visible=False,
            )
            download_zip = gr.DownloadButton(
                label="Скачать архив .zip",
                value=None,
                visible=False,
            )

    transcribe_button.click(
        fn=process_files,
        inputs=[files_input, language_dropdown, result_mode, punctuation_checkbox, speed_profile],
        outputs=[
            status_box,
            result_text,
            timestamps_text,
            result_table,
            result_files,
            download_zip,
            error_box,
        ],
        show_progress="full",
    )

    stop_button.click(
        fn=request_cancel,
        inputs=[],
        outputs=[status_box],
        queue=False,
        show_progress="hidden",
    )

    change_output_button.click(
        fn=change_output_directory,
        inputs=[],
        outputs=[status_box, output_path_box],
        queue=False,
        show_progress="hidden",
    )

    clear_button.click(
        fn=clear_ui,
        inputs=[],
        outputs=[
            status_box,
            result_text,
            timestamps_text,
            result_table,
            result_files,
            download_zip,
            error_box,
        ],
        show_progress="hidden",
    )

demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        show_error=True,
        allowed_paths=[str(OUTPUT_DIR)],
        css=custom_css,
    )
