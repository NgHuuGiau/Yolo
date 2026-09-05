# Lệnh Nhanh

Bảng lệnh sử dụng hàng ngày. Nếu đã hiểu hệ thống, đây là file tra cứu nhanh nhất.

---

## 1. Kiểm tra môi trường

```powershell
python run_doctor.py --skip-camera-check
python run_smoke.py
python run_smoke.py --ci-safe --stop-on-fail
python -m unittest discover -s tests -p "test_*.py"
```

---

## 2. Chat AI

### Desktop

```powershell
python run_chat.py --check-only
python run_chat.py --check-only --auto-fix-icons
python run_chat.py
python run_chat.py --cleanup-output --older-than-days 30
```

### Web

```powershell
python web_app.py
# hoặc
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
# Mở http://127.0.0.1:8000
# Admin DB: http://127.0.0.1:8000/admin/db
```

---

## 3. Camera realtime

```powershell
python run_app.py --advisor-only
python run_app.py
python run_app.py --mode medium
python run_app.py --camera-index 1
python run_app.py --model models/trained/best.pt
```

---

## 4. Medical CLI

```powershell
python run_medical.py status
python run_medical.py ready
python run_medical.py sources
python run_medical.py cancer
python run_medical.py analyze --image path/to/ảnh.jpg --patient-code BN001
```

---

## 5. Model & phân tích (Trạng thái 09/2026)

Hệ thống chỉ phân tích ảnh bằng model đã train sẵn (đặt trong `models/pretrained/`):

```powershell
# Kiểm tra đã đủ model chưa
python run_doctor.py --skip-camera-check
```

| Model | File | Trạng thái | Mô tả |
|---|---|---|---|
| **Brain** | `brain_classifier.pt` | ✅ **Sẵn sàng** | 4 loại u não (glioma/meningioma/pituitary/no_tumor) — fallback tự động |
| **Modality** | `modality_classifier.pt` | ✅ **99.93%** | 8 loại ảnh (CT, MRI, X-quang, Mammogram, Nội soi, Siêu âm, PET/CT, EUS) |
| **7 Ung thư** | `medical_7_cancers_cnn.pt` | ❌ **Chưa có** | Gan, phổi, vú, dạ dày, đại trực tràng, tiền liệt, tử cung — chờ data train mới |

> **Hành vi hiện tại**: Thiếu model 7 ung thư → hệ thống tự fallback sang **brain model** cho mọi ảnh. `run_doctor.py` báo "chỉ phân tích được não". Đây là dự kiến.

---

## 6. Giải thích nhanh

| Lệnh | Mục đích |
|---|---|
| `run_doctor.py --skip-camera-check` | Quét tổng thể, không cần webcam |
| `run_smoke.py` | Kiểm tra nhanh chuỗi entrypoint |
| `run_smoke.py --ci-safe` | Kiểm tra nhẹ, phù hợp CI |
| `run_app.py --advisor-only` | Gợi ý runtime trước khi mở camera |
| `run_chat.py --check-only` | Kiểm tra chat UI và medical sẵn sàng |
| `run_chat.py --cleanup-output` | Dọn file output cũ |
| `run_medical.py analyze` | Phân tích 1 ảnh y khoa (ưu tiên não) |

---

## 7. Trình tự trên máy mới

```powershell
python run_menu.py
python run_doctor.py --skip-camera-check
python run_smoke.py --ci-safe --stop-on-fail
python run_app.py --advisor-only
python run_chat.py --check-only
```

---

## 8. Khi có dữ liệu train mới (model 7 ung thư / brain tốt hơn)

1. Đặt file `.pt` vào `models/pretrained/`
2. Chạy `python run_doctor.py --skip-camera-check` để xác nhận nhận diện
3. `python run_medical.py analyze --image ...` để test