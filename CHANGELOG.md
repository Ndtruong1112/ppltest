# Changelog

Tất cả những thay đổi đáng chú ý trong dự án này sẽ được ghi lại trong file này.

---

## [Unreleased]

### Added
- ✨ GitHub Actions workflow cho BLEU score evaluation
- ✨ GitHub Actions workflow cho model training pipeline
- ✨ Requirements.txt cho quản lý dependencies
- ✨ CONTRIBUTING.md guide
- ✨ LICENSE file (MIT)
- ✨ CHANGELOG.md (file này)

### Fixed
- 🐛 Sửa lỗi encoding UTF-8 trên PowerShell
- 🐛 Sửa lỗi `eval_strategy` parameter (thay cho `evaluation_strategy`)
- 🐛 Sửa lỗi `tokenizer` parameter trong `Seq2SeqTrainer`
- 🐛 Loại bỏ FP16 khi chạy trên CPU
- 🐛 Thêm `# -*- coding: utf-8 -*-` cho các file Python

### Changed
- 📝 Cập nhật README.md với hướng dẫn mới
- 🔧 Điều chỉnh hyperparameters cho GPU training:
  - batch_size: 16 (từ 8)
  - num_epochs: 3 (từ 1)
  - Bật FP16 để tiết kiệm VRAM

---

## [v0.1.0] - 2025-05-15

### Initial Release

#### Added
- 📊 **train2.py**: Script huấn luyện mô hình chính
  - Fine-tuning Helsinki-NLP/opus-mt-en-vi
  - Hỗ trợ GPU/CPU tự động
  - Tokenization song song với 4 processes

- 📈 **final1.py**: Script đánh giá BLEU score
  - Tính BLEU score trên test set
  - Hiển thị chi tiết metrics (precision, brevity penalty, etc.)
  - In ví dụ dịch

- 🔄 **testdich.py**: Script test dịch thuật
  - Hỗ trợ inference trên câu tiếng Anh chuyên ngành IT
  - Output: dịch sang Tiếng Việt

- 📚 **train1.py**: Script thử nghiệm tokenization
  - Chuẩn bị dữ liệu cho training

- 📖 **README.md**: Tài liệu chi tiết dự án
  - Cấu trúc project
  - Hướng dẫn sử dụng
  - Cách cài đặt dependencies

---

## 🎯 Roadmap

### Phase 1 (Current)
- ✅ Setup project structure
- ✅ Implement basic training pipeline
- ✅ Add evaluation metrics
- ⏳ Add GitHub Actions CI/CD

### Phase 2 (Planned)
- [ ] Increase training data size (10k → 100k)
- [ ] Implement data augmentation
- [ ] Add validation set
- [ ] Fine-tune hyperparameters
- [ ] Improve BLEU score > 30

### Phase 3 (Future)
- [ ] Ensemble models
- [ ] Domain adaptation techniques
- [ ] API deployment
- [ ] Web interface
- [ ] Mobile app

---

## 📝 Conventions

### Commit Types
- `[Feature]` - Tính năng mới
- `[Fix]` - Sửa lỗi
- `[Docs]` - Cập nhật documentation
- `[Refactor]` - Tái cấu trúc code
- `[Test]` - Thêm tests
- `[Chore]` - Maintenance tasks

### Version Format
```
MAJOR.MINOR.PATCH
- MAJOR: Thay đổi breaking
- MINOR: Tính năng mới (backward compatible)
- PATCH: Bug fixes
```

---

**Được maintain bởi:** Ndtruong1112  
**Được cập nhật lần cuối:** May 15, 2025
