# Hệ Thống Dịch Máy IT Anh-Việt Bằng Fine-tuning PhoMT

## 1. Tổng quan đề tài

Đây là dự án fine-tuning mô hình dịch máy để dịch câu và tài liệu từ tiếng Anh sang tiếng Việt, đồng thời đánh giá mức cải thiện trước và sau quá trình tinh chỉnh.

Ý tưởng chính:
- Không train mô hình từ đầu.
- Sử dụng mô hình có sẵn `Helsinki-NLP/opus-mt-en-vi`.
- Fine-tune tiếp trên tập dữ liệu PhoMT để mô hình dịch đúng ngữ cảnh hơn.

## 2. Mục tiêu dự án

Mục tiêu của dự án là:
- Nhận câu tiếng Anh chuyên ngành IT.
- Sinh ra câu tiếng Việt tương ứng.
- Đánh giá chất lượng bằng BLEU score.
- So sánh với mô hình gốc hoặc với các mô hình phổ biến.
## 3. Công nghệ sử dụng

- Python
- PyTorch
- Hugging Face Transformers
- Datasets
- Evaluate / SacreBLEU
- GPU NVIDIA với CUDA

## 4. Cấu trúc dự án

Project code nằm ở:
- `C:\Users\Admin\ppltest`

Dữ liệu, model, checkpoint, cache nên đặt ở:
- `D:\Users\Admin\Downloads\PhoMT`

Các file chính:
- `train2.py`: script train chính
- `testdich.py`: script dịch thử
- `final1.py`: script đánh giá nhanh mô hình đã fine-tune
- `testmodelorigin.py`: script đánh giá mô hình gốc
- `compare_before_after.py`: script so sánh mô hình gốc và mô hình đã fine-tune
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
11. Đánh giá mô hình sau fine-tune bằng `final1.py`
12. So sánh mô hình gốc và mô hình đã fine-tune bằng `compare_before_after.py`

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
cd C:\Users\Admin\ppltest
$env:PHOMT_VENV_DIR = "D:\Users\Admin\Downloads\PhoMT\.venv"
$env:PHOMT_WORK_ROOT = "D:\Users\Admin\Downloads\PhoMT"
.\run_training.ps1
```

### Cách 2: Tạo venv mới hoàn toàn trên ổ D

```powershell
cd C:\Users\Admin\ppltest
.\setup_env_d.ps1
```

Sau đó:

```powershell
$env:PHOMT_VENV_DIR = "D:\Users\Admin\venvs\ppltest-gpu"
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

Dùng để đánh giá nhanh mô hình đã fine-tune.

Nó:
- Nạp tập test
- Dùng mô hình đã train để dịch toàn bộ tập test
- So sánh với câu dịch chuẩn
- Tính BLEU score trên một số lượng mẫu có thể cấu hình

### `testmodelorigin.py`

Dùng để đánh giá mô hình gốc `Helsinki-NLP/opus-mt-en-vi`.

Nó:
- Nạp mô hình gốc chưa fine-tune
- Dịch tập test
- Tính BLEU, chrF++, TER và thời gian suy luận

### `compare_before_after.py`

Đây là file quan trọng cho phần báo cáo và thuyết trình.

Nó:
- Chạy mô hình gốc và mô hình đã fine-tune trên cùng một tập test
- So sánh BLEU, chrF++, TER, thời gian dịch
- In ra ví dụ trước/sau để đánh giá định tính

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
- Batch size được đặt nhỏ và an toàn hơn cho GPU 6 GB
- Bật `fp16` để train nhanh hơn (giảm 1 nửa tiêu thụ vram so với fp32)
- Bật `pin_memory` để tăng tốc độ nạp dữ liệu (cố định vùng nhớ cho dữ liệu chuyển thẳng sang vram)
- Dùng `gradient_accumulation_steps` để bù lại khi batch size phải giảm vì giới hạn VRAM

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

Lưu ý với RTX 4050 6GB:
- Cấu hình hiện tại đang dùng `per_device_train_batch_size=2`
- `per_device_eval_batch_size=2`
- `gradient_accumulation_steps=8`
- Nếu vẫn hết VRAM, hãy đóng bớt ứng dụng đang dùng GPU hoặc giảm số lượng sample train để test trước

Test dịch:

```powershell
python testdich.py
```

Đánh giá BLEU:

```powershell
python final1.py
```

Đánh giá mô hình gốc:

```powershell
python testmodelorigin.py
```

So sánh trước và sau fine-tune:

```powershell
python compare_before_after.py
```

## 13. Cách điều chỉnh khi chạy trên máy khác

Nếu chạy dự án này trên máy khác, phần quan trọng nhất cần điều chỉnh là cấu hình trong `train2.py`.

### Các tham số cần quan tâm

Trong `train2.py`, các tham số ảnh hưởng trực tiếp đến khả năng train trên từng máy là:

- `per_device_train_batch_size`
- `per_device_eval_batch_size`
- `gradient_accumulation_steps`
- `dataloader_num_workers`
- số lượng mẫu train/test đang lấy ra
- `fp16`

### Ý nghĩa từng tham số

#### `per_device_train_batch_size`

Đây là số mẫu mà GPU xử lý trong một lần.

- Batch size càng lớn thì train có thể nhanh hơn
- Nhưng càng tốn VRAM
- Nếu quá lớn sẽ gây lỗi `CUDA out of memory`

Gợi ý:
- GPU 4 GB: thử từ `1`
- GPU 6 GB: thử từ `2`
- GPU 8 GB: thử từ `4`
- GPU 10-12 GB: thử từ `8`
- GPU 16 GB trở lên: có thể thử `16` hoặc cao hơn

#### `per_device_eval_batch_size`

Đây là batch size khi đánh giá trong quá trình train.

- Thường nên để bằng hoặc nhỏ hơn `per_device_train_batch_size`
- Nếu đánh giá bị tràn VRAM thì giảm tham số này trước

#### `gradient_accumulation_steps`

Tham số này giúp mô phỏng batch lớn hơn khi GPU không đủ VRAM.

Ví dụ:
- batch size thật = `2`
- `gradient_accumulation_steps = 8`

thì có thể hiểu gần đúng là mô hình tích lũy gradient như một batch lớn hơn, nhưng không cần nạp quá nhiều dữ liệu cùng lúc lên GPU.

Quy tắc đơn giản:
- GPU yếu hơn -> giảm `batch_size`, tăng `gradient_accumulation_steps`
- GPU mạnh hơn -> tăng `batch_size`, có thể giảm `gradient_accumulation_steps`

#### `dataloader_num_workers`

Tham số này quyết định số tiến trình phụ để nạp dữ liệu.

- CPU yếu hoặc máy dễ treo: để `0` hoặc `1`
- CPU khá: để `2`
- CPU mạnh hơn: có thể thử `4`

Nếu máy bị lag mạnh, treo hoặc dùng RAM quá nhiều, hãy giảm tham số này.

#### Số lượng mẫu train/test

Trong code hiện tại, dữ liệu đang được cắt ra để train thử nhanh hơn.

Ví dụ trong `train2.py`:
- train đang lấy `10000` mẫu
- test đang lấy `1000` mẫu

Nếu máy yếu hoặc chỉ muốn test pipeline, có thể giảm xuống:
- train: `1000` hoặc `2000`
- test: `200` hoặc `500`

Nếu máy mạnh hơn và bạn muốn train nghiêm túc hơn, có thể tăng số lượng mẫu.

#### `fp16`

`fp16` giúp:
- giảm tiêu thụ VRAM
- tăng tốc train trên GPU phù hợp

Thông thường:
- có GPU NVIDIA hỗ trợ tốt CUDA: nên bật
- nếu gặp lỗi lạ liên quan đến mixed precision: có thể thử tắt tạm để debug

### Cách chỉnh theo từng loại máy

#### Máy yếu, GPU ít VRAM

Ví dụ: 4 GB đến 6 GB VRAM

Nên ưu tiên:
- `per_device_train_batch_size=1` hoặc `2`
- `per_device_eval_batch_size=1` hoặc `2`
- `gradient_accumulation_steps=8` hoặc `16`
- `dataloader_num_workers=0` hoặc `1`
- giảm số mẫu train để test trước

#### Máy tầm trung

Ví dụ: 6 GB đến 8 GB VRAM

Nên thử:
- `per_device_train_batch_size=2` hoặc `4`
- `per_device_eval_batch_size=2` hoặc `4`
- `gradient_accumulation_steps=4` hoặc `8`
- `dataloader_num_workers=2`

Đây là nhóm máy gần với RTX 4050 Laptop 6 GB hiện tại của bạn.

#### Máy mạnh hơn

Ví dụ: 10 GB, 12 GB, 16 GB VRAM trở lên

Có thể thử:
- `per_device_train_batch_size=8`, `16`
- `per_device_eval_batch_size=8`, `16`
- `gradient_accumulation_steps=1`, `2`, hoặc `4`
- `dataloader_num_workers=2` hoặc `4`

Nhưng vẫn nên tăng từ từ, không tăng quá mạnh ngay từ đầu.

### Dấu hiệu để biết cần chỉnh lại

Nếu gặp các tình huống sau, bạn nên giảm cấu hình:

- báo lỗi `CUDA out of memory`
- màn hình chớp đen khi train
- máy quá lag khi đang train
- GPU full VRAM liên tục rồi crash
- Windows reset driver GPU

Khi đó nên làm theo thứ tự:

1. Giảm `per_device_train_batch_size`
2. Giảm `per_device_eval_batch_size`
3. Tăng `gradient_accumulation_steps`
4. Giảm `dataloader_num_workers`
5. Giảm số lượng mẫu train/test

### Cấu hình hiện tại phù hợp với máy của bạn

Máy hiện tại:
- GPU: `NVIDIA GeForce RTX 4050 Laptop GPU`
- VRAM: `6 GB`

Cấu hình đang đặt:
- `per_device_train_batch_size=2`
- `per_device_eval_batch_size=2`
- `gradient_accumulation_steps=8`
- `dataloader_num_workers=2`

Đây là mức an toàn để chạy thử trên máy hiện tại. Nếu vẫn tràn VRAM, hãy hạ tiếp batch size xuống `1`.

## 14. Kết luận

Phiên bản hiện tại thống nhất theo hướng:
- Code ở `C`
- venv, data, cache, model output ở `D`
- Training ưu tiên GPU
- Đánh giá và demo tách riêng, để trình bày
