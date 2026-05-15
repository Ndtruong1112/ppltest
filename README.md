# He Thong Dich May IT Anh-Viet Bang Fine-tuning PhoMT

## 1. Tong quan de tai

Day la du an fine-tuning mo hinh dich may de dich tai lieu chuyen nganh IT tu tieng Anh sang tieng Viet.

Y tuong chinh:
- Khong train mo hinh tu dau.
- Su dung mo hinh co san `Helsinki-NLP/opus-mt-en-vi`.
- Fine-tune tiep tren tap du lieu chuyen nganh IT de mo hinh dich dung ngu canh ky thuat hon.

Noi ngan gon de thuyet trinh:
"De tai cua em tap trung vao viec chuyen mot mo hinh dich tong quat thanh mo hinh dich chuyen biet cho linh vuc IT bang ky thuat fine-tuning."

## 2. Muc tieu du an

Muc tieu cua du an la:
- Nhan cau tieng Anh chuyen nganh IT.
- Sinh ra cau tieng Viet tuong ung.
- Danh gia chat luong bang BLEU score.
- Tan dung GPU de giam thoi gian huan luyen.

## 3. Cong nghe su dung

- Python
- PyTorch
- Hugging Face Transformers
- Datasets
- Evaluate / SacreBLEU
- GPU NVIDIA voi CUDA

## 4. Cau truc du an

Project code nam o:
- `C:\Users\Admin\pplnckh-1`

Du lieu, model, checkpoint, cache nen dat o:
- `D:\Users\Admin\Downloads\PhoMT`

Cac file chinh:
- `train2.py`: script train chinh
- `testdich.py`: script dich thu
- `final1.py`: script danh gia BLEU
- `project_config.py`: quan ly duong dan va cache
- `run_training.ps1`: script chay train de xac nhan GPU
- `setup_env_d.ps1`: tao virtualenv GPU tren o D

## 5. Luong xu ly da thong nhat

Luong chuan cua du an hien tai la:

1. Tao moi truong Python tren o `D`
2. Cai PyTorch ban ho tro CUDA
3. Cai cac thu vien trong `requirements.txt`
4. Doc du lieu song ngu tu thu muc `detokenization`
5. Tokenize du lieu
6. Nap mo hinh goc `Helsinki-NLP/opus-mt-en-vi`
7. Fine-tune mo hinh tren GPU
8. Luu checkpoint vao `results/`
9. Luu mo hinh cuoi vao `final_model_it/`
10. Test dich bang `testdich.py`
11. Danh gia BLEU bang `final1.py`

Noi ngan gon de thuyet trinh:
"Pipeline cua he thong gom 3 pha: chuan bi moi truong, huan luyen mo hinh, va danh gia ket qua."

## 6. Vi sao phai dua moi truong sang o D

Ly do:
- O `C` thuong dung cho he dieu hanh, de day bo nho.
- Model, cache va checkpoint cua Hugging Face kha lon.
- Khi train tren GPU, qua trinh tai model, luu checkpoint, tao cache co the ton nhieu dung luong.
- Dat env, cache va output tren `D` se de quan ly hon va tranh lam nang o he thong.

Trong phien ban da sua:
- Virtualenv co the dat tren o `D`
- Cache Hugging Face, PyTorch, pip, temp deu co the day sang `D`
- Model output va checkpoint mac dinh deu o `D`

## 7. Cach chay de tai dung chuan

### Cach 1: Dung env da co san tren o D

Neu ban da co env GPU o:
- `D:\Users\Admin\Downloads\PhoMT\.venv`

Thi chay:

```powershell
cd C:\Users\Admin\pplnckh-1
$env:PHOMT_VENV_DIR = "D:\Users\Admin\Downloads\PhoMT\.venv"
$env:PHOMT_WORK_ROOT = "D:\Users\Admin\Downloads\PhoMT"
.\run_training.ps1
```

### Cach 2: Tao env moi hoan toan tren o D

```powershell
cd C:\Users\Admin\pplnckh-1
.\setup_env_d.ps1
```

Sau do:

```powershell
$env:PHOMT_VENV_DIR = "D:\Users\Admin\venvs\pplnckh-1-gpu"
$env:PHOMT_WORK_ROOT = "D:\Users\Admin\Downloads\PhoMT"
.\run_training.ps1
```

## 8. Du lieu dau vao

Thu muc du lieu:
- `D:\Users\Admin\Downloads\PhoMT\detokenization`

Cau truc mong doi:

```text
detokenization/
|-- train/
|   |-- train.en
|   `-- train.vi
`-- test/
    |-- test.en
    `-- test.vi
```

Quy tac du lieu:
- Moi dong trong `train.en` phai ung voi 1 dong trong `train.vi`
- So dong 2 file phai bang nhau
- Encoding nen la UTF-8

Trong code moi, neu so dong khong khop, chuong trinh se bao loi ro rang.

## 9. Mo ta tung file theo cach de hieu

### `train2.py`

Day la file quan trong nhat.

No lam cac viec sau:
- Kiem tra GPU co san khong
- Doc du lieu train/test
- Tokenize cau Anh-Viet
- Nap mo hinh goc
- Fine-tune mo hinh
- Luu checkpoint va mo hinh cuoi

Noi de thuyet trinh:
"train2.py la trai tim cua he thong, vi no thuc hien toan bo quy trinh huan luyen tu du lieu thuan van ban den mo hinh da fine-tune."

### `testdich.py`

Dung de demo.

No:
- Nap mo hinh da train
- Nhap mot so cau tieng Anh
- Sinh ban dich tieng Viet

Day la file rat hop de demo tren slide hoac luc bao ve.

### `final1.py`

Dung de danh gia mo hinh.

No:
- Nap tap test
- Dung mo hinh da train de dich toan bo tap test
- So sanh voi cau dich chuan
- Tinh BLEU score

Noi de thuyet trinh:
"BLEU score giup do muc do gan dung giua cau dich cua mo hinh va cau tham chieu."

### `project_config.py`

File nay dung de thong nhat duong dan.

No giup:
- Khong can hard-code duong dan o nhieu noi
- Chuyen cache, temp, model output sang o `D`
- De sua duong dan hon neu doi may

## 10. GPU dang duoc dung nhu the nao

May cua ban da duoc kiem tra va co the nhan GPU:
- `NVIDIA GeForce RTX 4050 Laptop GPU`
- PyTorch CUDA dang hoat dong trong env o `D`

Trong `train2.py`, neu `torch.cuda.is_available()` la `True` thi:
- Model duoc dua len GPU
- Batch size duoc tang cao hon
- Bat `fp16` de train nhanh hon
- Bat `pin_memory` de tang toc do nap du lieu

Noi de thuyet trinh:
"GPU khong lam thay doi thuat toan, nhung giup giam manh thoi gian huan luyen nho kha nang tinh toan song song."

## 11. Nhung loi da duoc phat hien va xu ly

Truoc khi sua, du an co mot so van de:

1. Env trong project o `C` khong co `torch`, nen chay la loi ngay.
2. Code nam o `C` nhung du lieu va model lai hard-code sang `D`, gay roi luong.
3. Cache co nguy co do ve o `C`, de ton bo nho he thong.
4. Mot so file huong dan chua thong nhat voi luong chay thuc te.
5. `train1.py` con du code cu de gay hieu nham.

Sau khi sua:
- Luong da thong nhat hon
- README va quick-start phu hop hon voi code
- Script train uu tien env o `D`
- Cac thu muc cache/output duoc dua ve `D`

## 12. Cach giai thich de tai cho hoi dong

Ban co the noi theo khung nay:

### Mo dau

"De tai cua em la xay dung he thong dich may Anh-Viet cho linh vuc IT. Thay vi huan luyen mo hinh tu dau, em su dung mo hinh dich co san va fine-tune lai tren du lieu chuyen nganh."

### Du lieu

"Du lieu dau vao gom cac cap cau song ngu Anh-Viet. Moi cau tieng Anh se tuong ung voi mot cau tieng Viet. Du lieu duoc chia thanh tap train va tap test."

### Mo hinh

"Mo hinh nen em su dung la Helsinki-NLP/opus-mt-en-vi. Sau do em fine-tune de mo hinh hoc tot hon cac thuat ngu va ngu canh trong linh vuc cong nghe thong tin."

### Huong xu ly

"Quy trinh gom doc du lieu, tokenize, huan luyen tren GPU, luu mo hinh, test dich va danh gia bang BLEU score."

### Ket qua

"Ket qua duoc danh gia bang BLEU score va co the demo truc tiep bang cach nhap cau tieng Anh IT de xem cau dich tieng Viet."

## 13. Cach tu tin tra loi neu bi hoi

Neu bi hoi "Vi sao dung fine-tuning?"
- "Vi fine-tuning tiet kiem tai nguyen va thoi gian hon so voi train tu dau, dong thoi van giu duoc kien thuc ngon ngu tu mo hinh goc."

Neu bi hoi "Vi sao can GPU?"
- "Vi huan luyen mo hinh seq2seq rat ton tinh toan. GPU giup rut ngan thoi gian train dang ke."

Neu bi hoi "Vi sao can BLEU?"
- "Vi BLEU la mot chi so pho bien de danh gia chat luong dich may thong qua muc do trung khop voi ban dich tham chieu."

Neu bi hoi "Vi sao dua env sang o D?"
- "Vi dung luong model, cache va checkpoint lon; dua sang o D giup de quan ly va tranh lam day o he thong C."

## 14. Lenh hay dung

Tao env GPU tren o D:

```powershell
.\setup_env_d.ps1
```

Train:

```powershell
.\run_training.ps1
```

Test dich:

```powershell
python testdich.py
```

Danh gia BLEU:

```powershell
python final1.py
```

## 15. Ket luan

Phien ban hien tai da thong nhat luong chay theo huong:
- Code o `C`
- Env, data, cache, model output o `D`
- Training uu tien GPU
- Danh gia va demo tach rieng, de trinh bay

Neu ban dung dung luong nay, de tai se de giai thich hon, de demo hon, va it gap loi moi truong hon so voi phien ban cu.
# ppltest
