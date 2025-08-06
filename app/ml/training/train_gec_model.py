import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

MODEL_CHECKPOINT = "t5-small"
DATASET_NAME = "jfleg"
PREFIX = "fix-grammar: "

MODEL_OUTPUT_DIR = "app/ml/models/gec-t5-small-final"


def train_model():
    raw_datasets = load_dataset(DATASET_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def preprocess_function(examples):
        inputs = [PREFIX + sent for sent in examples["sentence"]]
        targets = [cor[0] for cor in examples["corrections"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        labels = tokenizer(text_target=targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    print("Tải model thành công!")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=5,
        num_train_epochs=3,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["validation"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(MODEL_OUTPUT_DIR)
    print("Đã lưu model thành công!")


if __name__ == "__main__":
    train_model()
    print("\nToàn bộ quá trình đã hoàn tất!")