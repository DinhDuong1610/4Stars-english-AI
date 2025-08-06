# AI-Powered Dictation Analysis API

Đây là một microservice backend được xây dựng bằng Python và FastAPI, chuyên cung cấp khả năng phân tích và chẩn đoán lỗi sai trong các bài nghe-chép chính tả tiếng Anh. API này nhận vào bài làm của người dùng và đáp án đúng, sau đó trả về điểm số, chi tiết so sánh, và quan trọng nhất là các giải thích lỗi sai được tạo ra bởi một mô hình AI tùy chỉnh.

## ✨ Tính năng Nổi bật

* **Chấm điểm & So sánh:** Cung cấp điểm số chính xác và hiển thị chi tiết sự khác biệt (thừa, thiếu, đúng) giữa hai đoạn văn bản.
* **Giải thích lỗi bằng AI:** Tích hợp một mô hình AI (fine-tuned T5-small) được huấn luyện để:
    * Xác định các loại lỗi ngữ pháp, chính tả, dấu câu...
    * Sinh ra một chuỗi giải thích có cấu trúc cho từng lỗi được phát hiện.
* **API hiệu suất cao:** Xây dựng trên nền tảng FastAPI và Uvicorn/Gunicorn, đảm bảo tốc độ xử lý nhanh và khả năng mở rộng.
* **Cấu trúc chuyên nghiệp:** Dự án được tổ chức theo từng lớp (API, Services, Schemas, ML Predictors) rõ ràng, dễ bảo trì và phát triển.

## 🛠️ Công nghệ Sử dụng

* **Backend:** Python, FastAPI, Gunicorn
* **Machine Learning / NLP:**
    * PyTorch
    * Hugging Face Transformers (cho model T5)
    * Hugging Face Datasets (để xử lý dữ liệu)
    * Errant (để gán nhãn lỗi tự động cho dữ liệu huấn luyện)
    * Spacy

## 🧠 Mô hình AI Tùy chỉnh (Custom AI Model)

Điểm nhấn của dự án là một mô hình AI được xây dựng tùy chỉnh để phân tích và giải thích lỗi sai, thay vì chỉ so sánh văn bản đơn thuần.

* **Kiến trúc (Architecture):** Mô hình được phát triển dựa trên kiến trúc **Transformer** nổi tiếng, cụ thể là **tinh chỉnh (fine-tuning)** mô hình **T5-small** (Text-to-Text Transfer Transformer) của Google.

* **Nhiệm vụ (Task):** Mô hình được huấn luyện cho một nhiệm vụ **Sequence-to-Sequence** chuyên biệt: **"Sinh giải thích lỗi có cấu trúc" (Structured Error Explanation Generation)**.
    * **Đầu vào:** Một chuỗi văn bản chứa cả câu sai của người dùng và câu đúng của đáp án.
    * **Đầu ra:** Một chuỗi văn bản có cấu trúc, liệt kê các lỗi đã được sửa và phân loại chúng.

* **Quy trình Xây dựng (Workflow):**
    1.  **Chuẩn bị Dữ liệu:** Tự động xử lý và gán nhãn cho **50,000 cặp câu** từ bộ dữ liệu **grammarly/coedit** bằng thư viện **errant** để tạo ra một bộ dữ liệu huấn luyện chất lượng cao.
    2.  **Huấn luyện (Fine-tuning):** Tinh chỉnh mô hình **t5-small** trên bộ dữ liệu đã được chuẩn bị để dạy cho nó khả năng nhận diện và mô tả các lỗi ngữ pháp.
    3.  **Tích hợp (Integration):** Đóng gói mô hình đã huấn luyện vào một module **"predictor"** và tích hợp trực tiếp vào business logic của API để cung cấp khả năng phân tích thời gian thực.

## 📂 Cấu trúc Dự án

Dự án được cấu trúc một cách khoa học để phân tách rõ ràng các thành phần:
```
dictation_api/
│
├── app/
│   ├── api/              # API endpoints (routers)
│   ├── ml/               # Machine Learning (training, predictors, models)
│   ├── schemas/          # Pydantic data models
│   └── services/         # Business logic
│
├── scripts/              # Các script phụ trợ
├── static/               # File tĩnh (audio...)
├── .gitignore
├── Dockerfile            # Công thức để đóng gói ứng dụng
├── main.py               # File khởi động chính
└── requirements.txt
```

## 🚀 API Endpoint

### `POST /api/v1/dictation/check`

Endpoint chính để phân tích một bài nghe-chép.

**Request Body:**

```json
{
  "user_text": "She dont has many informations.",
  "correct_text": "She doesn't have much information."
}
```

**Success Response (200 OK):**

```json
{
  "score": 60.53,
  "diffs": [
    { "type": "equal", "text": "she do" },
    { "type": "insert", "text": "nt" },
    { "type": "delete", "text": "esn't" },
    { "type": "equal", "text": " ha" },
    { "type": "insert", "text": "s" },
    { "type": "delete", "text": "ve much" },
    { "type": "equal", "text": " in" },
    { "type": "delete", "text": "form" },
    { "type": "insert", "text": "formation" },
    { "type": "equal", "text": "s." }
  ],
  "explanations": [
    "Tại 'dont': Lỗi khác (Nên sửa thành 'doesn't')",
    "Tại 'has': Lỗi dùng sai động từ (Nên sửa thành 'have')",
    "Tại 'many': Lỗi dùng sai từ hạn định (Nên sửa thành 'much')",
    "Tại 'informations': Lỗi số ít/số nhiều danh từ (Nên sửa thành 'information')"
  ]
}
```

## 🏁 Hướng dẫn Cài đặt và Chạy

**Yêu cầu:**
* Git
* Python 3.10+

**Các bước cài đặt:**

1.  **Clone repository:**
    ```bash
    git clone https://github.com/DinhDuong1610/4Stars-english-AI.git
    ```

2.  **Tạo và kích hoạt môi trường ảo:**
    ```bash
    # Tạo venv
    python -m venv venv
    # Kích hoạt venv (trên Windows)
    .\venv\Scripts\activate
    ```

3.  **Cài đặt các thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Tải mô hình ngôn ngữ cho `errant`:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **(Tùy chọn) Tự huấn luyện model AI:**
    * *Lưu ý: Các model đã được huấn luyện không được lưu trong repository này. Bạn cần tự huấn luyện chúng. Quá trình này rất tốn thời gian (nhiều giờ) và tài nguyên máy tính.*
    * **Bước 1: Chuẩn bị dữ liệu:**
        ```bash
        python app/ml/training/prepare_explanation_data.py
        ```
    * **Bước 2: Huấn luyện model:**
        ```bash
        python app/ml/training/train_explanation_generator.py
        ```

6.  **Chạy server:**
    ```bash
    uvicorn main:app --reload
    ```
    Ứng dụng sẽ chạy tại `http://127.0.0.1:8000`.
    Truy cập `http://127.0.0.1:8000/docs` để xem tài liệu và thử nghiệm API.

## 🌐 Dự án Liên quan (Related Project)
API này được xây dựng để phục vụ như một microservice chuyên biệt cho backend chính của website học tiếng Anh. Backend chính được xây dựng bằng Java Spring Boot.

➡️ Link tới dự án Java: https://github.com/DinhDuong1610/4stars-english-BE

## 👤 Tác giả

* Dương Đính
* jenny.180820@gmail.com