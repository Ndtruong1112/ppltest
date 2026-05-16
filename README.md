# Hệ Thống Dịch Máy IT Anh-Việt Bằng Fine-tuning PhoMT

## 1. Tổng quan đề tài

Đây là dự án fine-tuning mô hình dịch máy để dịch tài liệu chuyên ngành IT từ tiếng Anh sang tiếng Việt.

Ý tưởng chính:
- Không train mô hình từ đầu.
- Sử dụng mô hình có sẵn `Helsinki-NLP/opus-mt-en-vi`.
- Fine-tune tiếp trên tập dữ liệu chuyên ngành IT để mô hình dịch đúng ngữ cảnh kỹ thuật hơn.

Nói ngắn gọn để thuyết trình:
"Đề tài của em tập trung vào việc chuyển một mô hình dịch tổng quát thành mô hình dịch chuyên biệt cho lĩnh vực IT bằng kỹ thuật fine-tuning."

## 2. Mục tiêu dự án

Mục tiêu của dự án là:
- Nhận câu tiếng Anh chuyên ngành IT.
- Sinh ra câu tiếng Việt tương ứng.
- Đánh giá chất lượng bằng BLEU score.
- Tận dụng GPU để giảm thời gian huấn luyện.

## 3. Công nghệ sử dụng

- Python
- PyTorch
- Hugging Face Transformers
- Datasets
- Evaluate / SacreBLEU
- GPU NVIDIA với CUDA

## 4. Cấu trúc dự án

Project code nằm ở:
- `C:\Users\Admin\pplnckh-1`

Dữ liệu, model, checkpoint, cache nên đặt ở:
- `D:\Users\Admin\Downloads\PhoMT`

Các file chính:
- `train2.py`: script train chính
- `testdich.py`: script dịch thử
- `final1.py`: script đánh giá BLEU
- `project_config.py`: quản lý đường dẫn và cache
- `run_training.ps1`: script chạy train để xác nhận GPU
- `setup_env_d.ps1`: tạo virtualenv GPU trên ổ D

## 5. Luồng xử lý đã thống nhất

Luồng chuẩn của dự án hiện tại là:

1. Tạo môi trường Python trên ổ `D`
2. Cài PyTorch bản hỗ trợ CUDA
3. Cài các thư viện trong `requirements.txt`
4. Đọc dữ liệu song ngữ từ thư mục `detokenization`
5. Tokenize dữ liệu
6. Nạp mô hình gốc `Helsinki-NLP/opus-mt-en-vi`
7. Fine-tune mô hình trên GPU
8. Lưu checkpoint vào `results/`
9. Lưu mô hình cuối vào `final_model_it/`
10. Test dịch bằng `testdich.py`
11. Đánh giá BLEU bằng `final1.py`

Nói ngắn gọn để thuyết trình:
"Pipeline của hệ thống gồm 3 pha: chuẩn bị môi trường, huấn luyện mô hình, và đánh giá kết quả."

## 6. Vì sao phải đưa môi trường sang ổ D

Lý do:
- Ổ `C` thường dùng cho hệ điều hành, dễ đầy bộ nhớ.
- Model, cache và checkpoint của Hugging Face khá lớn.
- Khi train trên GPU, quá trình tải model, lưu checkpoint, tạo cache có thể tốn nhiều dung lượng.
- Đặt venv, cache và output trên `D` sẽ dễ quản lý hơn và tránh làm nặng ổ hệ thống.

Trong phiên bản đã sửa:
- Virtualenv có thể đặt trên ổ `D`
- Cache Hugging Face, PyTorch, pip, temp đều có thể đẩy sang `D`
- Model output và checkpoint mặc định đều ở `D`

## 7. Cách chạy đề tài đúng chuẩn

### Cách 1: Dùng venv đã có sẵn trên ổ D

Nếu bạn đã có venv GPU ở:
- `D:\Users\Admin\Downloads\PhoMT\.venv`

Thì chạy:

```powershell
cd C:\Users\Admin\pplnckh-1
$env:PHOMT_VENV_DIR = "D:\Users\Admin\Downloads\PhoMT\.venv"
$env:PHOMT_WORK_ROOT = "D:\Users\Admin\Downloads\PhoMT"
.\run_training.ps1 
```
### cách 2: Tạo venv mới hoàn toàn trên ổ D

```powershell
cd C:\Users\Admin\pplnckh-1
.\setup_env_d.ps1
```

Sau đó:

```powershell
$env:PHOMT_VENV_DIR = "D:\Users\Admin\venvs\pplnckh-1-gpu"
$env:PHOMT_WORK_ROOT = "D:\Users\Admin\Downloads\PhoMT"
.\run_training.ps1
```

## 8. Dữ liệu đầu vào

Thư mục dữ liệu:
- `D:\Users\Admin\Downloads\PhoMT\detokenization`

Cấu trúc mong đợi:

```text
detokenization/
|-- train/
|   |-- train.en
|   `-- train.vi
`-- test/
    |-- test.en
    `-- test.vi
```

Quy tắc dữ liệu:
- Mỗi dòng trong `train.en` phải ứng với 1 dòng trong `train.vi`
- Số dòng 2 file phải bằng nhau
- Encoding nên là UTF-8

Trong code mới, nếu số dòng không khớp, chương trình sẽ báo lỗi rõ ràng.

## 9. Mô tả từng file theo cách dễ hiểu

### `train2.py`

Đây là file quan trọng nhất.

Nó làm các việc sau:
- Kiểm tra GPU có sẵn không
- Đọc dữ liệu train/test
- Tokenize câu Anh-Viet
- Nạp mô hình gốc
- Fine-tune mô hình
- Lưu checkpoint và mô hình cuối

"train2.py là trái tim của hệ thống, vì nó thực hiện toàn bộ quy trình huấn luyện từ dữ liệu thuần văn bản đến mô hình đã fine-tune."

### `testdich.py`

Dùng để demo.

Nó:
- Nạp mô hình đã train
- Nhập một số câu tiếng Anh
- Sinh bản dịch tiếng Việt

### `final1.py`

Dùng để đánh giá mô hình.

Nó:
- Nạp tập test
- Dùng mô hình đã train để dịch toàn bộ tập test
- So sánh với câu dịch chuẩn
- Tính BLEU score

### `project_config.py`

File này dùng để thống nhất đường dẫn.

Nó giúp:
- Không cần Hard-code đường dẫn ở nhiều nơi
- Chuyển cache, temp, model output sang ổ `D`
- Dễ sửa đường dẫn hơn nếu đổi máy

## 10. GPU đang được dùng thế nào

Máy tính đã được kiểm tra và có thể nhận GPU:
- `NVIDIA GeForce RTX 4050 Laptop GPU`
- PyTorch CUDA đang hoạt động trong venv ở `D`

Trong `train2.py`, nếu `torch.cuda.is_available()` là `True` thì:
- Model được đưa lên GPU
- Batch size được tăng cao hơn(hiểu đơn giản là kích thước data để mô hình tiếp nhận 1 lần)
- Bật `fp16` để train nhanh hơn (giảm 1 nửa tiêu thụ vram so với fp32)
- Bật `pin_memory` để tăng tốc độ nạp dữ liệu (cố định vùng nhớ cho dữ liệu chuyển thẳng sang vram)

"GPU không làm thay đổi thuật toán, nhưng giúp giảm mạnh thời gian huấn luyện nhờ khả năng tính toán song song."

## 11. Những lỗi đã được phát hiện và xử lý

Trước khi sửa, dự án có 1 số vấn đề:

1. Venv trong project ở `C` không có `torch`, nên chạy là lỗi ngay.
2. Code nằm ở `C` nhưng dữ liệu và model lại hard-code sang `D`, gây rối luồng.
3. Cache có nguy cơ đổ về ổ `C`, dễ tốn bộ nhớ hệ thống.
4. Một số file hướng dẫn chưa thống nhất với lượng chạy thực tế.
5. `train1.py` còn dư code cũ dễ gây hiểu nhầm.

Sau khi sửa:
- Luồng đã thống nhất hơn
- README va quick-start phù hợp hơn với code
- Script train ưu tiên venv ở `D`
- Các thư mục cache/output được đưa về `D`

## 12. Lệnh hay dùng

Tạo venv GPU trên ổ D:

```powershell
.\setup_env_d.ps1
```

Train:

```powershell
.\run_training.ps1
```

Test dịch:

```powershell
python testdich.py
```

Đánh giá BLEU:

```powershell
python final1.py
```

## 13. Kết luận

Phiên bản hiện tại thống nhất theo hướng:
- Code ở `C`
- venv, data, cache, model output ở `D`
- Training ưu tiên GPU
- Đánh giá và demo tách riêng, để trình bày

# ppltest
