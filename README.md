# OncoVision

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![Windows](https://img.shields.io/badge/Windows-11%2B-0078D6)](https://www.microsoft.com/windows)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111111)](https://www.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)](https://opencv.org/)
[![PySide6](https://img.shields.io/badge/PySide6-Desktop%20UI-41CD52)](https://www.qt.io/qt-for-python)
[![FastAPI](https://img.shields.io/badge/FastAPI-Web%20Chat-009688)](https://fastapi.tiangolo.com/)
[![Medical](https://img.shields.io/badge/Medical-Screening-00A6A6)](docs/medical_imaging_guide.md)
[![Training](https://img.shields.io/badge/Training-YOLO-FFB000)](docs/training_guide.md)

**OncoVision** là nền tảng hỗ trợ chẩn đoán hình ảnh y khoa tích hợp: từ quản lý dữ liệu y tế, huấn luyện mô hình YOLO/CNN đến giao diện chat AI cho bác sĩ và nhân viên y tế. Hệ thống chạy hoàn toàn trên máy local (Windows), hỗ trợ xử lý đa dạng định dạng ảnh y khoa (DICOM, NIfTI, JPG, PNG).

---

## Trạng thái hiện tại (09/2026)

| Thành phần | Trạng thái |
|---|---|
| **Modality Classifier** | ✅ Sẵn sàng — ResNet18, 8 loại ảnh, **99.93% acc** (test 5,640 ảnh) |
| **Brain Classifier** | ✅ Sẵn sàng — ConvNeXt-Tiny, 4 loại u não (glioma/meningioma/pituitary/no_tumor), **fallback tự động** khi thiếu model tổng |
| **Model 7 ung thư** | ❌ Chưa có (`medical_7_cancers_cnn.pt` — đang chờ dữ liệu train mới) |
| **Web UI (FastAPI)** | ✅ Chạy được — `python web_app.py` → http://127.0.0.1:8000 |
| **Desktop Chat (PySide6)** | ✅ Chạy được — `python run_chat.py` |
| **Test suite** | ✅ **293 tests pass** |

> **Lưu ý**: Hệ thống hiện chỉ phân tích được **ung thư não** (model chính). Các nhóm ung thư khác cần model `medical_7_cancers_cnn.pt` — sẽ bổ sung khi có dữ liệu train mới.

---

## Tính năng chính

| Nhóm | Mô tả |
|---|---|
| **Chat AI Y khoa** | Giao diện desktop (PySide6) và web (FastAPI) để đặt câu hỏi, tải ảnh y khoa và nhận phân tích tự động |
| **Phân tích ảnh y tế** | **Ưu tiên não** (4 loại u) — modality tự động nhận diện; 7 nhóm khác chờ model |
| **Camera thông minh** | Chạy realtime object detection với nhiều chế độ (auto/high/medium/low), tự động gợi ý cấu hình runtime phù hợp với máy |
| **Huấn luyện mô hình** | Pipeline train YOLO detection và CNN classifier đầy đủ, hỗ trợ resume, augment dữ liệu, export model |

---

## Bắt đầu nhanh

### Yêu cầu hệ thống

- Windows 10/11
- Python 3.10+
- GPU NVIDIA khuyến nghị cho train và inference

### Cài đặt

```powershell
git clone <repo-url>
cd OncoVision
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Database

Hệ thống dùng **SQLite** tập trung tại `output/onco.db` — lưu hội thoại chat + case y tế trong 1 file. Không cần cài DB server.

### Kiểm tra môi trường

```powershell
python run_doctor.py --skip-camera-check
python run_smoke.py
```

### Chạy chat AI

```powershell
# Giao diện desktop
python run_chat.py

# Giao diện web (optional)
python web_app.py
# hoặc
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

---

## Bản đồ entrypoint

| Entrypoint | Vai trò |
|---|---|
| `run_chat.py` | Giao diện chat AI desktop — kiểm tra trạng thái, mở chat, dọn dẹp output |
| `run_app.py` | Camera realtime — gợi ý cấu hình runtime, chạy object detection trực tiếp |
| `run_menu.py` | Menu tổng hợp, cửa vào cho người vận hành |
| `run_doctor.py` | Quét tổng thể hệ thống — dependency, model, dataset, output |
| `run_medical.py` | CLI quản lý luồng y dược — dataset, model, phân tích, modality, báo cáo |
| `run_smoke.py` | Kiểm tra nhanh chuỗi entrypoint (CI-friendly) |
| `run_tests.py` | Dashboard chạy unit test |
| `web_app.py` | Web app upload ảnh → nhận diện → phân tích (FastAPI) |

### Luồng dữ liệu cơ bản

```
Camera:  run_app.py → core/camera_runner.py → output/captures/
Chat:    run_chat.py → app/chat_ui/ → medical/phân tích → output/chat/
Medical: run_medical.py → medical/dataset.py → output/medical/
Model:   models/pretrained/*.pt (bổ sung file model đã train sẵn)
Web:     web_app.py → SQLite → output/onco.db
```

---

## Model & phân tích

Hệ thống **chỉ phân tích ảnh** bằng các model đã train sẵn — không tự huấn luyện:

```powershell
# Đặt các file model đã train vào models/pretrained/:
#   brain_classifier.pt        → Não (4 sub-label) — **ĐÃ CÓ, SẴN SÀNG**
#   modality_classifier.pt     → Modality (8 loại ảnh y tế) — **ĐÃ CÓ, 99.93%**
#   medical_7_cancers_cnn.pt   → 7 ung thư (gan, phổi, vú, dạ dày, đại trực tràng, tiền liệt, tử cung) — **CHƯA CÓ, CHỜ DATA MỚI**

# Kiểm tra hệ thống đã nhận đủ model chưa
python run_doctor.py --skip-camera-check

# Phân tích 1 ảnh (ưu tiên não)
python run_medical.py analyze --image path/to/ảnh.jpg --patient-code BN001
```

> Model `medical_7_cancers_cnn.pt` chưa có → `run_doctor.py` sẽ báo "chỉ phân tích được não". Đây là hành vi dự kiến.

---

## Cấu trúc thư mục

```text
OncoVision/
├── app/                    # Giao diện và runtime
│   ├── camera_runtime/     # Bootstrap và launch camera
│   └── chat_ui/            # Chat desktop, theme, storage, widgets
├── core/                   # Xử lý camera, model loader, hardware info
├── medical/                # Luồng y dược — dataset, model, pipeline, chat, report
├── training/               # Model catalog & download models (dùng cho menu)
├── utils/                  # Helper dùng chung
├── config/                 # Cấu hình YAML
├── dataset/                # Dữ liệu
│   ├── medical/            # Dataset y tế
│   │   └── Ung thư não/    # Raw/processed cho 4 loại u não
│   ├── medical_modality/   # Dataset modality (8 loại)
│   └── object_detection/   # Dataset detection
├── models/                 # Mô hình
│   ├── pretrained/         # Model tiền huấn luyện (brain, modality, yolo11*)
│   └── trained/            # Model đã train (trống — chờ data mới)
├── output/                 # Kết quả đầu ra
├── docs/                   # Tài liệu
└── tests/                  # Unit test
```

---

## Medical Pipeline (OncoVision AI)

Hệ thống phân tích ảnh y khoa với CNN classifier (convnext_tiny pretrained @512px cho não, resnet18 @320px cho modality):

| Bước | Module | Mô tả |
|---|---|---|
| Validate | `medical/validator.py` | Kiểm tra ảnh đầu vào, modality, body region |
| DICOM parse | `medical/dataset.py` | Parse DICOM header, window/level rendering |
| CNN inference | `medical/cnn_classifier.py` | FP16 trên GPU, TTA, confidence calibration |
| Grad-CAM | `medical/explainability.py` | Heatmap vùng CNN tập trung |
| Chat UI | `app/chat_ui/` | Desktop (PySide6) + Web (FastAPI) |
| Báo cáo | `medical/reporting.py` | JSON/MD/HTML dashboard |

## Tài liệu tham khảo

| File | Nội dung |
|---|---|
| [docs/project_overview.md](docs/project_overview.md) | Tổng quan kiến trúc và cây thư mục |
| [docs/install_guide.md](docs/install_guide.md) | Hướng dẫn cài đặt chi tiết |
| [docs/medical_imaging_guide.md](docs/medical_imaging_guide.md) | Luồng y dược — dataset, model, training |
| [docs/training_guide.md](docs/training_guide.md) | Huấn luyện object detection YOLO |
| [docs/runtime_tool_guide.md](docs/runtime_tool_guide.md) | Runtime advisor và camera realtime |
| [docs/quick_commands.md](docs/quick_commands.md) | Lệnh nhanh hàng ngày |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Lỗi thường gặp và cách xử lý |
| [docs/ci_and_quality.md](docs/ci_and_quality.md) | CI pipeline và quality gate |