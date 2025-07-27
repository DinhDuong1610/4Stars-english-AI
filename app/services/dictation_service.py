import diff_match_patch as dmp_module
from typing import Dict, List
import re
from app.ml.predictors.explanation_generator import generate_explanation

ERROR_TYPE_EXPLANATIONS = {
    "R:ADJ": "Lỗi dùng sai tính từ", "R:ADV": "Lỗi dùng sai trạng từ",
    "R:CONJ": "Lỗi dùng sai liên từ", "R:DET": "Lỗi dùng sai từ hạn định (a, an, the...)",
    "R:NOUN": "Lỗi dùng sai danh từ", "R:NOUN:NUM": "Lỗi số ít/số nhiều danh từ",
    "R:NOUN:POSS": "Lỗi sở hữu cách", "R:ORTH": "Lỗi viết hoa/viết thường",
    "R:PART": "Lỗi dùng sai tiểu từ", "R:PREP": "Lỗi dùng sai giới từ",
    "R:PRON": "Lỗi dùng sai đại từ", "R:PUNCT": "Lỗi dấu câu",
    "R:SPELL": "Lỗi chính tả", "R:VERB": "Lỗi dùng sai động từ",
    "R:VERB:FORM": "Lỗi sai dạng của động từ", "R:VERB:SVA": "Lỗi hoà hợp chủ vị",
    "R:VERB:TENSE": "Lỗi chia thì của động từ",
    "M:ADJ": "Lỗi thiếu tính từ", "M:ADV": "Lỗi thiếu trạng từ",
    "M:CONJ": "Lỗi thiếu liên từ", "M:DET": "Lỗi thiếu từ hạn định",
    "M:NOUN": "Lỗi thiếu danh từ", "M:PART": "Lỗi thiếu tiểu từ",
    "M:PREP": "Lỗi thiếu giới từ", "M:PRON": "Lỗi thiếu đại từ",
    "M:PUNCT": "Lỗi thiếu dấu câu", "M:VERB": "Lỗi thiếu động từ",
    "U:ADJ": "Lỗi thừa tính từ", "U:ADV": "Lỗi thừa trạng từ",
    "U:CONJ": "Lỗi thừa liên từ", "U:DET": "Lỗi thừa từ hạn định",
    "U:NOUN": "Lỗi thừa danh từ", "U:PART": "Lỗi thừa tiểu từ",
    "U:PREP": "Lỗi thừa giới từ", "U:PRON": "Lỗi thừa đại từ",
    "U:PUNCT": "Lỗi thừa dấu câu", "U:VERB": "Lỗi thừa động từ",
    "MORPH": "Lỗi hình thái từ", "WO": "Lỗi sai trật tự từ",
    "R:OTHER": "Lỗi", "U:OTHER": "Lỗi thừa từ", "M:OTHER": "Lỗi thiếu từ"
}


def parse_explanation_string(explanation: str) -> List[str]:
    if not explanation:
        return []

    parts = re.findall(r'\[(.*?)]', explanation)
    results = []
    for part in parts:
        try:
            change, error_code = part.split('|')
            original, corrected = change.split('->')

            error_code = error_code.strip()
            error_explanation = ERROR_TYPE_EXPLANATIONS.get(error_code, f"Lỗi khác (mã: {error_code})")

            results.append(f"Tại '{original.strip()}': {error_explanation} (Nên sửa thành '{corrected.strip()}')")
        except ValueError:
            continue
    return results


def check_dictation_answer(user_text: str, correct_text: str) -> Dict:
    explanation_string = generate_explanation(user_text, correct_text)
    explanations = parse_explanation_string(explanation_string)

    dmp = dmp_module.diff_match_patch()
    lower_user_text = user_text.lower()
    lower_correct_text = correct_text.lower()
    diffs_raw = dmp.diff_main(lower_correct_text, lower_user_text)
    dmp.diff_cleanupSemantic(diffs_raw)

    diffs_processed = []
    correct_char_count = 0
    total_char_count = len(lower_correct_text)

    for diff_type, text in diffs_raw:
        diff_item = {"type": "equal", "text": text}
        if diff_type == 0:
            correct_char_count += len(text)
        elif diff_type == -1:
            diff_item["type"] = "delete"
        elif diff_type == 1:
            diff_item["type"] = "insert"
        diffs_processed.append(diff_item)

    score = (correct_char_count / total_char_count) * 100 if total_char_count > 0 else 0

    return {
        "score": round(score, 2),
        "diffs": diffs_processed,
        "explanations": explanations
    }