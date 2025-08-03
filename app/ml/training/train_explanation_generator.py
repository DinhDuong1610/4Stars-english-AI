from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer

DATA_FILE = "app/ml/training/explanation_train_data.csv"
MODEL_CHECKPOINT = "t5-small"
PREFIX = "explain edits: "
MODEL_OUTPUT_DIR = "app/ml/models/explanation-generator-t5-small"


def train_explanation_model():
    print(f"Đang tải dữ liệu từ: {DATA_FILE}...")
    raw_datasets = load_dataset('csv', data_files=DATA_FILE)

    split_datasets = raw_datasets['train'].train_test_split(test_size=0.1, seed=42)
    print("Đã tải và chia bộ dữ liệu thành công!")
    print(split_datasets)

    print(f"Đang tải tokenizer cho model: {MODEL_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]

        model_inputs = tokenizer(inputs, max_length=256, truncation=True)
        labels = tokenizer(text_target=targets, max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Bắt đầu tokenize dữ liệu...")
    tokenized_datasets = split_datasets.map(preprocess_function, batched=True)
    print("Tokenize dữ liệu thành công!")

    print("Bắt đầu tải model cơ sở T5...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,

        num_train_epochs=3,

        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        predict_with_generate=True,

        logging_strategy="steps",
        logging_steps=500,
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=3,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n" + "=" * 50)
    print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MODEL T5")
    print("Quá trình này có thể mất nhiều thời gian...")
    print("=" * 50 + "\n")

    trainer.train()

    print(f"\nHuấn luyện hoàn tất! Lưu model cuối cùng vào thư mục: {MODEL_OUTPUT_DIR}")
    trainer.save_model(MODEL_OUTPUT_DIR)
    print("Đã lưu model thành công!")


if __name__ == "__main__":
    train_explanation_model()