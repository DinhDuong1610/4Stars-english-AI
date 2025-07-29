import errant
from datasets import load_dataset
from tqdm import tqdm
import spacy
import csv

DATASET_NAME = "grammarly/coedit"
OUTPUT_FILE = "app/ml/training/explanation_train_data.csv"
NUM_SAMPLES = 50000


def clean_sentence(text: str) -> str:
    colon_index = text.find(':')
    if colon_index != -1:
        return text[colon_index + 1:].strip()
    return text.strip()

def create_explanation_data():

    print("Đang khởi tạo công cụ phân tích lỗi (errant)...")
    try:
        annotator = errant.load('en')
    except OSError:
        print("Đang tải mô hình spacy 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        annotator = errant.load('en')
    print("Khởi tạo thành công!")

    print(f"Đang tải {NUM_SAMPLES} mẫu từ bộ dữ liệu '{DATASET_NAME}'...")
    dataset_split = load_dataset(DATASET_NAME, split=f"train[:{NUM_SAMPLES}]")
    print("Tải dữ liệu thành công!")

    print(f"Bắt đầu xử lý dữ liệu và ghi ra file: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["input_text", "target_text"])

        for entry in tqdm(dataset_split):
            try:
                original_sentence = clean_sentence(entry["src"])
                corrected_sentence = entry["tgt"]

                if not original_sentence or not corrected_sentence or original_sentence == corrected_sentence:
                    continue

                orig_parsed = annotator.parse(original_sentence)
                cor_parsed = annotator.parse(corrected_sentence)
                edits = annotator.annotate(orig_parsed, cor_parsed)

                if not edits:
                    continue

                input_text = f"explain edits: incorrect: {original_sentence} correct: {corrected_sentence}"

                target_parts = []
                for edit in edits:
                    target_parts.append(f"[{edit.o_str} -> {edit.c_str} | {edit.type}]")
                target_text = ", ".join(target_parts)

                writer.writerow([input_text, target_text])

            except Exception as e:
                print(f"Bỏ qua một câu do lỗi: {e}")
                continue

    print(f"Xử lý hoàn tất! Dữ liệu huấn luyện đã được lưu tại {OUTPUT_FILE}")


if __name__ == "__main__":
    create_explanation_data()