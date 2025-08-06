import datasets
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, \
    TrainingArguments, Trainer
import numpy as np

DATA_FILE = "app/ml/training/train_bert_data.txt"
MODEL_CHECKPOINT = "bert-base-uncased"
MODEL_OUTPUT_DIR = "app/ml/models/error-analyzer-bert"

def train_analyzer_model():

    def read_data_file(file_path):
        tokens, tags = [], []
        all_tokens, all_tags = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    token, tag = line.split()
                    tokens.append(token)
                    tags.append(tag)
                else:
                    if tokens:
                        all_tokens.append(tokens)
                        all_tags.append(tags)
                    tokens, tags = [], []
        return all_tokens, all_tags

    all_tokens, all_tags = read_data_file(DATA_FILE)

    unique_tags = sorted(list(set(tag for tag_list in all_tags for tag in tag_list)))
    label_to_id = {tag: i for i, tag in enumerate(unique_tags)}
    id_to_label = {i: tag for i, tag in enumerate(unique_tags)}

    all_tags_as_ids = [[label_to_id[tag] for tag in tag_list] for tag_list in all_tags]

    dataset = DatasetDict({
        "train": datasets.Dataset.from_dict({"tokens": all_tokens, "ner_tags": all_tags_as_ids})
    }).shuffle(seed=42)["train"].train_test_split(test_size=0.1)

    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(unique_tags),
        id2label=id_to_label,
        label2id=label_to_id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(MODEL_OUTPUT_DIR)
    print("Đã lưu model thành công!")


if __name__ == "__main__":
    train_analyzer_model()