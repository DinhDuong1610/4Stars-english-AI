import errant
from datasets import load_dataset
from tqdm import tqdm
import spacy

DATASET_NAME = "grammarly/coedit"
OUTPUT_FILE = "app/ml/training/train_bert_data.txt"
NUM_SAMPLES = 50000


def create_training_data():
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
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in tqdm(dataset_split):
            try:
                original_sentence = entry["src"]
                corrected_sentence = entry["tgt"]

                orig_parsed = annotator.parse(original_sentence)
                cor_parsed = annotator.parse(corrected_sentence)
                edits = annotator.annotate(orig_parsed, cor_parsed)

                edit_tags = {edit.o_start: edit.type for edit in edits}

                for i, token in enumerate(orig_parsed):
                    tag = edit_tags.get(i, "C")
                    f.write(f"{token.text} {tag}\n")

                f.write("\n")
            except Exception as e:
                print(f"Bỏ qua một câu do lỗi: {e}")
                continue

    print(f"Xử lý hoàn tất! {NUM_SAMPLES} mẫu đã được xử lý và lưu tại {OUTPUT_FILE}")


if __name__ == "__main__":
    create_training_data()