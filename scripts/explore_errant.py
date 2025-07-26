import errant

def main():

    print("Đang khởi tạo công cụ phân tích lỗi (errant)...")
    annotator = errant.load('en')
    print("Khởi tạo thành công!")
    print("-" * 40)

    original_sentence = "He walk to the market."
    corrected_sentence = "He walks to the market."

    print(f"Câu gốc: '{original_sentence}'")
    print(f"Câu sửa: '{corrected_sentence}'")

    orig_parsed = annotator.parse(original_sentence)
    cor_parsed = annotator.parse(corrected_sentence)

    edits = annotator.annotate(orig_parsed, cor_parsed)

    print("\nCác lỗi được phát hiện:")
    if not edits:
        print("Không tìm thấy lỗi nào.")
    else:
        for edit in edits:
            print(f"-> Lỗi tại: '{edit.o_str}'")
            print(f"   Sửa thành: '{edit.c_str}'")
            print(f"   Loại lỗi: {edit.type}")

    print("-" * 40)

    original_sentence_2 = "I am writting a letter to my freind."
    corrected_sentence_2 = "I am writing a letter to my friend."

    print(f"Câu gốc: '{original_sentence_2}'")
    print(f"Câu sửa: '{corrected_sentence_2}'")

    orig_parsed_2 = annotator.parse(original_sentence_2)
    cor_parsed_2 = annotator.parse(corrected_sentence_2)
    edits_2 = annotator.annotate(orig_parsed_2, cor_parsed_2)

    print("\nCác lỗi được phát hiện:")
    if not edits_2:
        print("Không tìm thấy lỗi nào.")
    else:
        for edit in edits_2:
            print(f"-> Lỗi tại: '{edit.o_str}'")
            print(f"   Sửa thành: '{edit.c_str}'")
            print(f"   Loại lỗi: {edit.type}")


if __name__ == "__main__":
    main()