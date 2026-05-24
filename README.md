# 🎯 Hệ Thống Dịch Máy IT Anh-Việt Bằng Fine-tuning PhoMT

Dự án này thực hiện tối ưu hóa và tinh chỉnh (Fine-tuning) mô hình dịch máy dịch thuật chuyên ngành Công nghệ thông tin từ tiếng Anh sang tiếng Việt dựa trên mô hình pretrained `Helsinki-NLP/opus-mt-en-vi` và tập dữ liệu song ngữ chuyên ngành PhoMT.

Hệ thống đã được thiết kế và tối ưu hóa tối đa cho môi trường cục bộ trên máy tính chạy **Windows 11 (GPU RTX 4050 6GB VRAM, CPU Ryzen 7 8745H, 16GB RAM)**.

---

## 📖 Tổng Quan Đề Tài

Dịch máy chuyên ngành, đặc biệt là lĩnh vực Công nghệ Thông tin (IT), luôn là một thách thức lớn đối với các mô hình dịch máy thông thường do tính đặc thù cao của thuật ngữ và ngữ cảnh công nghệ (ví dụ: *pointer, heap, database schema, normalization, thread,...*). Các mô hình dịch thuật thương mại hoặc mô hình pretrained phổ thông thường dịch word-by-word hoặc dịch sai ý nghĩa kỹ thuật của các từ khóa này.

Đề tài này tập trung vào việc nghiên cứu và xây dựng hệ thống dịch máy chuyên biệt cho lĩnh vực IT từ tiếng Anh sang tiếng Việt. Bằng cách kế thừa mô hình nền tảng mạnh mẽ chuyên về ngôn ngữ Anh-Việt là **`Helsinki-NLP/opus-mt-en-vi`** và thực hiện quá trình tinh chỉnh chuyên biệt (**Fine-tuning**) trên tập dữ liệu song ngữ chất lượng cao **PhoMT**, hệ thống được kỳ vọng sẽ hiểu sâu sắc thuật ngữ IT và dịch chính xác, tự nhiên hơn.

Hệ thống được tối ưu hóa toàn diện cho việc chạy huấn luyện và đánh giá trên phần cứng cá nhân thông dụng (máy tính RTX 4050 Laptop 6GB VRAM), giúp giảm rào cản tài nguyên tính toán nhưng vẫn đạt hiệu suất cao nhất.

## 🎯 Mục Tiêu Dự Án

1. **Xây dựng bộ dịch thuật IT chuyên sâu**: Chuyển dịch chính xác các câu, tài liệu kỹ thuật chứa nhiều thuật ngữ IT từ tiếng Anh sang tiếng Việt phù hợp với văn phong công nghệ thông tin tại Việt Nam.
2. **Triển khai đa dạng kỹ thuật tinh chỉnh**:
   - **LoRA Fine-tuning (Parameter-Efficient)**: Cập nhật lượng nhỏ tham số giúp huấn luyện siêu tốc, tốn ít tài nguyên bộ nhớ VRAM.
   - **Full Fine-tuning (Toàn diện)**: Cập nhật toàn bộ trọng số của mô hình để đạt độ hội tụ và tối ưu hóa sâu nhất.
3. **Huấn luyện quy mô lớn trên máy cá nhân**: Đáp ứng mức dữ liệu huấn luyện lớn với mặc định **500,000 câu song ngữ** và **10,000 câu đánh giá** với kích thước batch size mặc định là **64** hoạt động ổn định trên GPU 6GB VRAM.
4. **Đánh giá & Kiểm thử khách quan**: Đo lường chất lượng dịch thuật thông qua các chỉ số tiêu chuẩn quốc tế như **BLEU, chrF++, TER** để so sánh trực quan hiệu quả của mô hình trước và sau khi fine-tune.
5. **Đơn giản hóa quy trình sử dụng**: Đóng gói toàn bộ mã nguồn huấn luyện, kiểm thử và demo giao diện dịch thử tương tác thành một mạch chạy thống nhất thông qua menu runner thông minh.

---

## 🚀 Tính Năng Nổi Bật & Tối Ưu Hóa Hiệu Năng

1. **Khởi Động Tức Thì (`dataloader_num_workers=0`):**
   * Triệt tiêu hoàn toàn thời gian chờ 1–2 phút ở bước chuẩn bị nạp dữ liệu do cơ chế `spawn` chậm của Windows. Hệ thống bắt đầu huấn luyện ngay lập tức trên luồng chính (Main Thread).
2. **Siêu Tốc Độ Đánh Giá Lúc Train (`predict_with_generate=False`):**
   * Quá trình đánh giá trung gian khi đang huấn luyện chỉ tính toán hàm Loss (Entropy chéo) song song trên GPU thay vì chạy dịch tự hồi quy (dịch từng từ) cho 10,000 câu. Thời gian eval giảm từ vài phút xuống **chưa đầy 1 giây** mỗi epoch.
3. **Mở Khóa Tối Đa Sức Mạnh GPU (`gradient_checkpointing=False`):**
   * Do mô hình `opus-mt-en-vi` rất nhỏ (~77M tham số), việc tắt gradient checkpointing giúp giảm khối lượng tính toán lại của GPU, tăng tốc độ xử lý thêm **25-30%** mà vẫn chỉ chiếm khoảng 3GB/6GB VRAM (an toàn tuyệt đối ở batch size 64).
4. **Trình Điều Khiển Nhất Quán (Unified Interactive Runner):**
   * Tích hợp toàn bộ quy trình: Huấn luyện Full, Huấn luyện LoRA, So sánh đánh giá BLEU, Dịch thử trực quan vào một file chạy duy nhất (`run_training.bat` hoặc `run_training.ps1`).

---

## 📁 Cấu Trúc Dự Án

Mã nguồn dự án được lưu trữ tại ổ `C:`, trong khi dữ liệu lớn, thư mục ảo (Virtualenv), mô hình lưu trữ và bộ đệm (cache) được định tuyến lưu trữ trên ổ `D:` để tiết kiệm dung lượng ổ hệ thống.

* **Project Code:** `C:\Users\Admin\ppltest`
* **Data, Cache & Models:** `D:\Users\Admin\Downloads\PhoMT`

### Chi tiết các file chính:
* `run_training.bat` / `run_training.ps1`: Trình chạy giao diện menu tương tác hợp nhất cho CMD và PowerShell.
* `train2.py`: Script huấn luyện Full Fine-Tuning (cập nhật toàn bộ tham số, đã tối ưu hóa).
* `train_lora.py`: Script huấn luyện Parameter-Efficient LoRA (PEFT, rất nhẹ và hiệu quả cao).
* `compare_before_after.py`: Đánh giá & so sánh song song BLEU, chrF++, TER giữa mô hình Gốc và mô hình Full Fine-tuned.
* `compare_lora.py`: Đánh giá & so sánh BLEU, chrF++, TER giữa mô hình Gốc và mô hình LoRA.
* `testdich.py`: Trình dịch thử tương tác nhập câu trực tiếp từ dòng lệnh.
* `project_config.py`: File cấu hình dùng chung để ánh xạ các thư mục làm việc, thư mục cache và output sang ổ `D:`.
* `setup_env_d.ps1`: Script PowerShell để tự động khởi tạo môi trường `.venv` tương thích GPU trên ổ `D:`.
* `train1.py`: *(Cũ - Đã ngưng sử dụng)* Script thử nghiệm đầu tiên.

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Quy Trình Hợp Nhất

### Bước 1: Chuẩn bị môi trường ảo GPU
Nếu chưa khởi tạo môi trường, hãy chạy file PowerShell sau để tự động tạo `.venv` trên ổ `D:` và cài đặt PyTorch với CUDA:
```powershell
# Chạy trên PowerShell (Quyền Admin nếu cần)
.\setup_env_d.ps1
```

### Bước 2: Chạy trình điều khiển hợp nhất
Bạn chỉ cần mở terminal và chạy file runner (không cần sửa code):
* Trên **Command Prompt (CMD)** hoặc double-click:
  ```cmd
  run_training.bat
  ```
* Trên **PowerShell**:
  ```powershell
  .\run_training.ps1
  ```

Một menu điều khiển trực quan sẽ xuất hiện:
```text
=========================================================
   PhoMT IT Translation Pipeline Runner (RTX 4050/6GB)   
=========================================================

📋 SELECT AN ACTION TO PERFORM:
  [1] Train: LoRA Fine-Tuning (Recommended: fast, fits 6GB VRAM easily)
  [2] Train: Full Fine-Tuning (Updates all parameters, uses more VRAM)
  [3] Eval: Compare Base model vs LoRA Adapter (BLEU/chrF++/TER)
  [4] Eval: Compare Base model vs Full Fine-Tuned (BLEU/chrF++/TER)
  [5] Test: Translate individual sentences interactively
  [6] Exit
```

---

## ⚙️ Các Biến Môi Trường Điều Khiển (Tùy Chọn Cao Cấp)

Các script Python đọc trực tiếp các tham số cấu hình thông qua biến môi trường. Bạn có thể thay đổi các giá trị này trước khi chạy:

| Biến Môi Trường | Mô Tả | Mặc Định |
| :--- | :--- | :--- |
| `PHOMT_BATCH_SIZE` | Batch size cho mỗi bước (cho cả train và eval) | `64` |
| `PHOMT_GRADIENT_ACCUMULATION_STEPS` | Số bước tích lũy gradient trước khi cập nhật trọng số | `2` |
| `PHOMT_TRAIN_SAMPLES` | Số câu train tối đa cho Full Fine-tuning | `500000` (hoặc `full`) |
| `PHOMT_EVAL_SAMPLES` | Số câu eval tối đa trong lúc train | `10000` (hoặc `full`) |
| `PHOMT_LORA_TRAIN_SAMPLES` | Số câu train tối đa cho LoRA | `500000` (hoặc `full`) |
| `PHOMT_LORA_EVAL_SAMPLES` | Số câu eval tối đa cho LoRA | `10000` (hoặc `full`) |
| `PHOMT_GRADIENT_CHECKPOINTING` | Bật/tắt tính năng checkpoint để tiết kiệm VRAM | `False` (để tối ưu tốc độ) |
| `PHOMT_NUM_WORKERS` | Số luồng CPU nạp dữ liệu | `0` (để tránh lỗi spawn trên Windows) |
| `PHOMT_EVAL_STRATEGY` | Chiến lược đánh giá trong lúc huấn luyện | `epoch` (tránh đánh giá theo step quá nhiều lần) |

Ví dụ: Nếu muốn thay đổi số lượng câu train thử nghiệm nhanh thành 5,000 câu trước khi train full:
* Trên **PowerShell**:
  ```powershell
  $env:PHOMT_LORA_TRAIN_SAMPLES = "5000"
  $env:PHOMT_LORA_EVAL_SAMPLES = "500"
  .\run_training.ps1
  ```
* Trên **Command Prompt (CMD)**:
  ```cmd
  set PHOMT_LORA_TRAIN_SAMPLES=5000
  set PHOMT_LORA_EVAL_SAMPLES=500
  run_training.bat
  ```

---

## 📊 Kết Quả Huấn Luyện & Đánh Giá Thực Tế (LoRA)

Dưới đây là bảng kết quả so sánh thu được sau khi thực hiện Fine-Tuning LoRA (chỉ thử nghiệm trên 1,000 dòng dữ liệu mẫu) trên tập test:

| Chỉ số | Mô hình Gốc (Pretrained) | Mô hình LoRA (Chỉ train 1000 câu) | Cải thiện |
| :--- | :---: | :---: | :---: |
| **BLEU Score** (Cao là tốt) | 23.38 | **28.85** | **+5.47** |
| **chrF++** | 44.97 | **50.70** | **+5.73** |
| **TER** (Thấp là tốt) | 64.64 | **58.73** | **-5.91** |

---

## ⚖️ Giấy Phép & Tài Liệu Tham Khảo

* Bộ dữ liệu: PhoMT (Bilingual Vietnamese-English Translation Dataset).
* Mô hình nền tảng: `Helsinki-NLP/opus-mt-en-vi` (Hugging Face).
