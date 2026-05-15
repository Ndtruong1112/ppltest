# Hướng Dẫn Đóng Góp Dự Án PhoMT IT Translation

## 🎯 Mục Đích Dự Án

Dự án này nhằm fine-tune mô hình dịch máy cho lĩnh vực IT (Anh → Việt).

---

## 📋 Quy Trình Đóng Góp

### 1. Fork & Clone Repository
```bash
git clone https://github.com/Ndtruong1112/pplnckh.git
cd pplnckh
```

### 2. Tạo Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### 4. Tạo Branch Mới
```bash
git checkout -b feature/your-feature-name
```

---

## 🚀 Các Task Có Thể Đóng Góp

### Training & Evaluation
- [ ] Tăng số lượng training data (10k → 100k)
- [ ] Thêm data augmentation
- [ ] Thử nghiệm các hyperparameter khác
- [ ] Thêm validation set riêng
- [ ] Ensemble với các mô hình khác

### Code Improvements
- [ ] Thêm error handling tốt hơn
- [ ] Thêm logging chi tiết
- [ ] Tối ưu hóa tốc độ training
- [ ] Thêm unit tests
- [ ] Cải thiện documentation

### Documentation
- [ ] Cập nhật README.md
- [ ] Thêm các ví dụ mới
- [ ] Viết blog post về dự án
- [ ] Cải thiện docstrings trong code

### Data & Quality
- [ ] Đóng góp dữ liệu IT chất lượng cao
- [ ] Xác minh chất lượng dịch
- [ ] Báo cáo lỗi dịch

---

## ✅ Checklist Trước Khi Submit

- [ ] Code chạy mà không lỗi
- [ ] Đã test locally
- [ ] Theo PEP 8 style guide
- [ ] Thêm comments/docstrings nếu cần
- [ ] Không thêm file lớn vào git
- [ ] Cập nhật README.md nếu cần

---

## 📝 Commit Message Format

```
[TYPE] Brief description

Detailed explanation if needed.

- Feature: New functionality
- Fix: Bug fixes
- Docs: Documentation only
- Refactor: Code restructuring
- Test: Adding tests
```

**Example:**
```
[Feature] Add data augmentation for training

- Implement back-translation augmentation
- Increase training data size by 2x
- Improve BLEU score by 1.5%
```

---

## 🤝 Pull Request Process

1. **Cập nhật main branch:**
   ```bash
   git pull origin main
   ```

2. **Push branch của bạn:**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Tạo Pull Request trên GitHub:**
   - Mô tả rõ thay đổi của bạn
   - Link đến issue liên quan (nếu có)
   - Attach screenshots/results nếu cần

4. **Chờ review:**
   - Maintainer sẽ review code
   - Nếu có feedback, hãy cập nhật
   - Merge khi được approve

---

## 📞 Liên Hệ & Hỗ Trợ

- **Issues:** Báo lỗi hoặc đề xuất tính năng
- **Discussions:** Thảo luận về ý tưởng
- **GitHub Discussions:** Hỏi đáp chung

---

## 📜 License

Dự án này sử dụng MIT License. Xem `LICENSE` file.

---

**Cảm ơn vì đóng góp! 🙏**
