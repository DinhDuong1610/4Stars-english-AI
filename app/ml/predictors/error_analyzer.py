import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "app/ml/models/error-analyzer-bert"

model = None
tokenizer = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        print(f"Đang load model phân tích lỗi từ: {MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)
        print("Load model thành công!")


def analyze_errors(text: str) -> list:
    _load_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids).logits

    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    results = []
    current_word = ""
    current_tag = ""

    for token, tag in zip(tokens, predicted_token_class):
        if token in [tokenizer.cls_token, tokenizer.sep_token]:
            continue

        if token.startswith("##"):
            current_word += token.replace("##", "")
        else:
            if current_word:
                results.append({"word": current_word, "tag": current_tag})

            # Bắt đầu một từ mới
            current_word = token
            current_tag = tag

    if current_word:
        results.append({"word": current_word, "tag": current_tag})

    return results


if __name__ == '__main__':
    sentence = "He are students"
    analysis = analyze_errors(sentence)

    print("-" * 30)
    print(f"Phân tích câu: '{sentence}'")
    for item in analysis:
        if item['tag'] != 'C':
            print(f"  - Từ '{item['word']}' có lỗi: {item['tag']}")
    print("-" * 30)