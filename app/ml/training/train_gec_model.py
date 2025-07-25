import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer

MODEL_CHECKPOINT = "t5-small"
DATASET_NAME = "jfleg"
PREFIX = "fix-grammar: "

def prepare_and_process_data():
    print("Bắt đầu tải dữ liệu và tokenizer...")

    raw_datasets = load_dataset(DATASET_NAME)
    print("\nCấu trúc dữ liệu thô:")
    print(raw_datasets)

    print("\nVí dụ dữ liệu:")
    print("Câu sai:", raw_datasets["validation"][0]["sentence"])
    print("Các câu sửa đúng:", raw_datasets["validation"][0]["corrections"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    print(f"\nĐã tải tokenizer cho model: {MODEL_CHECKPOINT}")

    def preprocess_function(examples):
        inputs = [PREFIX + sent for sent in examples["sentence"]]

        targets = [cor[0] for cor in examples["corrections"]]

        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\nBắt đầu tiền xử lý và tokenize dữ liệu...")
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    print("Xử lý dữ liệu thành công!")

    print("\nCấu trúc dữ liệu sau khi xử lý:")
    print(tokenized_datasets)
    print("\nVí dụ một bản ghi đã xử lý (input_ids):")
    print(tokenized_datasets["validation"][0]["input_ids"])

    return tokenizer, tokenized_datasets


if __name__ == "__main__":
    prepare_and_process_data()
    print("\nHoàn tất bước chuẩn bị dữ liệu!")