import torch
import os
import wave
import numpy as np
from transformers import pipeline

OUTPUT_DIR = "static/audio"
MODEL_ID = "facebook/mms-tts-eng"

TEXT_TO_GENERATE = {
    "exercise_1": "Hello world, this is a test.",
    "exercise_2": "The quick brown fox jumps over the lazy dog.",
    "exercise_3": "FastAPI is a modern, fast web framework for building APIs.",
    "exercise_4": "Natural Language Processing is a fascinating field of artificial intelligence."
}


def generate_audio_files():
    print("Bắt đầu quá trình tạo file audio với model Facebook MMS...")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Đã tạo thư mục: {OUTPUT_DIR}")

    print(f"Đang tải model MMS: {MODEL_ID}...")
    synthesiser = pipeline("text-to-speech", model=MODEL_ID)
    print("Tải model thành công!")

    for filename, text in TEXT_TO_GENERATE.items():
        output_path = os.path.join(OUTPUT_DIR, f"{filename}.wav")
        print(f"Đang xử lý: {filename}.wav")

        output = synthesiser(text)

        audio = output["audio"]
        sample_rate = output["sampling_rate"]

        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(output_path, 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(audio_int16.tobytes())

        print(f"-> Đã lưu thành công tại: {output_path}")

    print("\nHoàn tất! Tất cả các file đã được tạo.")


if __name__ == "__main__":
    generate_audio_files()