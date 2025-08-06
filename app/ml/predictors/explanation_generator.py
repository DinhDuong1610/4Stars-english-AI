import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "app/ml/models/explanation-generator-t5-small"

model = None
tokenizer = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        print(f"Đang load model sinh giải thích từ: {MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
        print("Load model thành công!")


def generate_explanation(incorrect_text: str, correct_text: str) -> str:
    _load_model()

    input_text = f"explain edits: incorrect: {incorrect_text} correct: {correct_text}"

    input_ids = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=128)

    explanation_string = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return explanation_string


if __name__ == '__main__':
    test_cases = [
        {
            "incorrect": "One of my friend live in the city.",
            "correct": "One of my friends lives in the city."
        },
        {
            "incorrect": "She go to the cinema yesteday.",
            "correct": "She went to the cinema yesterday."
        },
        {
            "incorrect": "I am interested on listen to a music.",
            "correct": "I am interested in listening to music."
        },
        {
            "incorrect": "Althought the wether were bad, they decide to going for a walk.",
            "correct": "Although the weather was bad, they decided to go for a walk."
        },
        {
            "incorrect": "Why he is here?",
            "correct": "Why is he here?"
        }
    ]

    print("=" * 50)
    print("BẮT ĐẦU KIỂM TRA HIỆU QUẢ MODEL")
    print("=" * 50)

    for i, case in enumerate(test_cases):
        incorrect = case["incorrect"]
        correct = case["correct"]

        explanation = generate_explanation(incorrect, correct)

        print(f"\n--- Ví dụ {i + 1} ---")
        print(f"Câu sai:    '{incorrect}'")
        print(f"Câu đúng:   '{correct}'")
        print(f"AI giải thích: {explanation}")

    print("\n" + "=" * 50)
    print("KIỂM TRA HOÀN TẤT")
    print("=" * 50)