import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "app/ml/models/gec-t5-small-final"
PREFIX = "fix-grammar: "

model = None
tokenizer = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def _load_model():

    global model, tokenizer

    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)

        print("Load model thành công!")


def correct_grammar(text: str) -> str:
    _load_model()

    input_text = PREFIX + text

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=128)

    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return corrected_text


if __name__ == '__main__':
    incorrect_sentence_1 = "He dont know nothing."
    corrected_sentence_1 = correct_grammar(incorrect_sentence_1)

    print("-" * 30)
    print(f"Câu sai:     '{incorrect_sentence_1}'")
    print(f"Câu sửa đúng: '{corrected_sentence_1}'")
    print("-" * 30)

    incorrect_sentence_2 = "we can not live if old people could not find siences"
    corrected_sentence_2 = correct_grammar(incorrect_sentence_2)
    print(f"Câu sai:     '{incorrect_sentence_2}'")
    print(f"Câu sửa đúng: '{corrected_sentence_2}'")
    print("-" * 30)