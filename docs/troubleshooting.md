# Xử Lý Sự Cố

Các lỗi thường gặp và hướng dẫn xử lý.

---

## 1. Camera không mở được

```powershell
python run_app.py --advisor-only
python run_doctor.py --skip-camera-check
python run_app.py --mode low --camera-index 1
```

Nếu vẫn lỗi:

- Kiểm tra ứng dụng khác có đang dùng webcam không
- Thử camera-index khác (0, 1, 2)
- Xem `core/camera_runner.py` để debug

---

## 2. Model không có sẵn

Kiểm tra:

- `models/pretrained/` — model tiền huấn luyện
- `models/trained/` — model đã train

Nếu cần tải pretrained: `python training/download_models.py`.

---

## 3. Chat UI chưa sẵn sàng

```powershell
python run_chat.py --check-only --auto-fix-icons
```

Nếu thất bại, xem:

- `utils/entrypoint_checks.py`
- `medical/system_status.py`
- `app/chat_ui/`

---

## 4. Model medical chưa sẵn sàng

Hệ thống chỉ phân tích bằng model đã train sẵn. Kiểm tra:

```powershell
python run_doctor.py --skip-camera-check
python run_medical.py status
```

**Trạng thái model (09/2026):**
- ✅ `brain_classifier.pt` — Não (4 loại u) — **SẴN SÀNG**
- ✅ `modality_classifier.pt` — 8 modality — **99.93%**
- ❌ `medical_7_cancers_cnn.pt` — 7 ung thư — **CHƯA CÓ**

Nếu thiếu model:
- Bổ sung file đã train vào `models/pretrained/`
- Hoặc sửa đường dẫn trong `config/medical_settings.yaml`
- Xem `medical/model_policy.py` để hiểu cách resolve đường dẫn model

> **Lưu ý**: Thiếu model 7 ung thư → hệ thống tự fallback sang brain model cho mọi ảnh. `run_doctor.py` báo "chỉ phân tích được não". Đây là hành vi dự kiến.

---

## 5. Medical status sai

Xem:

- `medical/system_status.py`
- `medical/model_policy.py`
- `medical/training.py`
- `run_medical.py`

---

## 6. CI fail

Xem theo thứ tự:

1. `.github/workflows/test.yml`
2. `run_smoke.py`
3. `ci-logs/04-ruff.txt`
4. `ci-logs/05-mypy-type-check.txt`
5. `ci-logs/07-smoke-check.txt`

---

## 7. Web Chat UI không mở được

```powershell
python web_app.py
# hoặc
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
# Mở http://127.0.0.1:8000
```

Nếu lỗi:

- Kiểm tra đã cài dependencies: `pip install -r requirements.txt`
- Kiểm tra port 8000 có bị chiếm: `netstat -ano | findstr :8000`
- Thử port khác: `python -m uvicorn web_app:app --host 127.0.0.1 --port 8080`
- Xem log server để tìm lỗi cụ thể
- Admin DB: `http://127.0.0.1:8000/admin/db`
