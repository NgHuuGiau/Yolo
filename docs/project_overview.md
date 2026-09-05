# Tổng Quan Kiến Trúc Dự Án

Tài liệu này mô tả kiến trúc tổng thể của OncoVision: cây thư mục, trách nhiệm từng module và luồng dữ liệu giữa các thành phần.

---

## 1. Tổng quan

OncoVision là monorepo gồm bốn nhánh chính:

| Nhánh | Vai trò |
|---|---|
| **Camera thông minh** | Chạy realtime object detection với YOLO, hỗ trợ nhiều chế độ runtime |
| **Y dược** | Phân tích ảnh bệnh lý bằng model CNN đã train sẵn (không tự train) |
| **Chat AI** | Giao diện desktop và web cho bác sĩ tương tác với hệ thống phân tích |
| **Phân tích model** | Nhận ảnh → nhận diện modality/vùng cơ thể → phân loại ung thư từ `models/pretrained/` |

---

## 2. Cây thư mục

```
OncoVision/
├── app/                     # Giao diện và runtime
│   ├── camera_runtime/      # Parser, bootstrap, launch camera
│   └── chat_ui/             # Chat window, storage, theme, widgets, medical controller
├── assets/                  # Tài nguyên tĩnh (icon, font)
├── config/                  # Cấu hình hệ thống (YAML)
├── core/                    # Xử lý camera cốt lõi
│   ├── camera_runner.py     # Vòng lặp camera: detect, overlay, record, capture
│   ├── model_loader.py      # Nạp YOLO model và fallback
│   ├── hardware_info.py     # Đọc CPU/GPU/CUDA/PyTorch
│   ├── frame_processing.py  # Tiền xử lý frame
│   ├── runtime_advisor.py   # Gợi ý cấu hình runtime
│   └── tracking/            # Gán track, smooth, filter detection
├── dataset/                 # Dữ liệu vận hành
│   ├── medical/             # Dataset y tế (ung thư não: glioma/meningioma/pituitary/no_tumor)
│   ├── medical_modality/    # Dataset phân loại modality (8 loại)
│   └── object_detection/    # Dataset YOLO detection
├── docs/                    # Tài liệu
├── medical/                 # Logic nghiệp vụ y dược
│   ├── dataset.py           # Tạo và kiểm tra layout dataset
│   ├── system_status.py     # Tổng hợp trạng thái medical
│   ├── training.py          # Audit, split, train, validate
│   ├── output_management.py # Quản lý output medical
│   ├── storage.py           # Lưu và truy vấn case DB
│   ├── validator.py         # Kiểm tra ảnh đầu vào
│   └── cli_helpers.py       # Helper in trạng thái CLI
├── models/                  # Mô hình
│   ├── pretrained/          # YOLO pretrained, modality classifier, brain classifier
│   └── trained/             # Model đã train (best.pt)
├── output/                  # Kết quả đầu ra
│   ├── captures/            # Ảnh chụp từ camera
│   ├── chat/                # File chat capture
│   ├── medical/             # Kết quả phân tích y tế
│   └── recordings/          # Video ghi từ camera
├── scripts/                 # Script tiện ích (đang trống)
├── tests/                   # Unit test
├── training/                # Model catalog & download models
├── utils/                   # Helper dùng chung
│   ├── entrypoint_checks.py # Kiểm tra trạng thái entrypoint
│   ├── cleanup_utils.py     # Dọn dẹp output
│   ├── console_ui.py        # UI bảng trong console
│   ├── file_utils.py        # Xử lý file
│   ├── logger.py            # Ghi log
│   ├── camera_utils.py      # Tiện ích camera
│   └── sqlite_utils.py      # Tiện ích SQLite
├── web_app.py               # Giao diện chat web (FastAPI)
├── run_chat.py              # Entrypoint chat
├── run_app.py               # Entrypoint camera
├── run_menu.py              # Menu tổng hợp
├── run_doctor.py            # Quét hệ thống
├── run_medical.py           # CLI y dược
└── run_smoke.py             # Kiểm tra CI
```

---

## 3. Entrypoint

Tất cả entrypoint đều là lớp mỏng: gọi module xử lý tương ứng và trả về kết quả.

| File | Trách nhiệm |
|---|---|
| `run_chat.py` | Khởi chạy chat UI desktop, kiểm tra trạng thái preflight, dọn output |
| `run_app.py` | Runtime advisor và camera realtime |
| `run_menu.py` | Cửa vào tổng hợp cho người vận hành |
| `run_doctor.py` | Quét tổng thể hệ thống |
| `run_medical.py` | CLI quản lý nhánh y dược |
| `run_smoke.py` | Smoke check (CI-friendly) |
| `run_tests.py` | Dashboard chạy unit test |
| `web_app.py` | Web app upload ảnh → phân tích (FastAPI) |

---

## 4. Luồng dữ liệu

### Camera realtime

```
run_app.py
→ app/camera_runtime/
→ core/hardware_info.py
→ core/runtime_advisor.py
→ core/model_loader.py
→ core/camera_runner.py
→ output/captures/ | output/recordings/
```

### Y dược

```
dataset/medical/
→ medical/dataset.py
→ medical/system_status.py
→ run_medical.py (CLI)
→ output/medical/
→ run_chat.py (Chat UI)
```

### Phân tích model

```
models/pretrained/*.pt (model đã train sẵn)
→ medical/pipeline.py (MedicalImageAnalyzer)
→ medical/cnn_classifier.py | medical/explainability.py
→ output/medical/ (report JSON/MD/HTML)
→ web_app.py | run_chat.py (Chat UI)
```

### Chat AI

```
run_chat.py
→ app/chat_ui/ (window, storage, widgets)
→ medical/ (phân tích, case DB)
→ output/chat/ | output/medical/
→ SQLite (lịch sử)
```

---

## 5. Debug theo triệu chứng

| Vấn đề | File cần mở đầu tiên |
|---|---|
| Camera không chạy | `run_app.py`, `core/camera_runner.py`, `utils/camera_utils.py` |
| Runtime gợi ý sai | `core/hardware_info.py`, `core/runtime_advisor.py` |
| Chat UI không sẵn sàng | `run_chat.py`, `utils/entrypoint_checks.py`, `app/chat_ui/` |
| Medical status sai | `medical/system_status.py`, `medical/model_policy.py` |
| Model chưa sẵn sàng | `medical/model_policy.py`, `config/medical_settings.yaml` |
| CI fail | `.github/workflows/test.yml`, `run_smoke.py` |
