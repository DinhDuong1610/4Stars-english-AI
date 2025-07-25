import diff_match_patch as dmp_module
from typing import List, Dict

def check_dictation_answer(user_text: str, correct_text: str) -> Dict:
    dmp = dmp_module.diff_match_patch()

    diffs_raw = dmp.diff_main(user_text.lower(), correct_text.lower())
    dmp.diff_cleanupSemantic(diffs_raw)

    diffs_processed = []
    correct_char_count = 0
    total_char_count = len(correct_text)

    for diff_type, text in diffs_raw:
        # diff_type: -1 (delete), 0 (equal), 1 (insert)
        if diff_type == 0:
            diffs_processed.append({"type": "equal", "text": text})
            correct_char_count += len(text)
        elif diff_type == -1:
            diffs_processed.append({"type": "delete", "text": text})
        elif diff_type == 1:
            diffs_processed.append({"type": "insert", "text": text})

    score = (correct_char_count / total_char_count) * 100 if total_char_count > 0 else 0

    return {"score": round(score, 2), "diffs": diffs_processed}