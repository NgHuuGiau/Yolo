# Hướng Dẫn Luồng Y Dược

Tài liệu giải thích luồng y tế của OncoVision: dataset, entrypoint, lệnh CLI, cấu hình training và debug.

---

## 1. Tổng quan

Nhánh y dược phục vụ bốn việc chính:

1. Quản lý dataset y tế (tổ chức, kiểm tra, báo cáo)
2. Theo dõi trạng thái model và output medical
3. Phân tích ảnh y khoa và lưu trữ ca bệnh
4. Hỗ trợ chat AI kiểm tra trạng thái y dược

---

## 2. Entrypoint liên quan

| File | Vai trò |
|---|---|
| `run_medical.py` | CLI chính — dataset, model, training, modality |
| `run_chat.py` | Chat UI — kiểm tra preflight, tích hợp medical pipeline |
| `run_doctor.py` | Quét tổng thể model, dataset, output |

---

## 3. Lệnh CLI

```powershell
python run_medical.py status          # Trạng thái tổng quan
python run_medical.py ready           # Kiểm tra đủ điều kiện phân tích
python run_medical.py sources         # Liệt kê nguồn ảnh
python run_medical.py cancer          # Danh sách nhóm ung thư
python run_medical.py init-dataset    # Khởi tạo layout dataset
```

Hệ thống **chỉ phân tích ảnh** bằng model đã train sẵn trong `models/pretrained/`. Các lệnh `train*` chỉ dùng khi bạn tự huấn luyện model bên ngoài rồi đặt file kết quả vào `models/pretrained/`.

Lưu ý: `init-dataset` chỉ in layout mong đợi, không tự tạo dữ liệu.

---

## 4. Nhóm bệnh và modality hỗ trợ (Trạng thái 09/2026)

| Nhóm | Ảnh / volume thường dùng | Trạng thái model |
|---|---|---|
| **Não** | MRI, CT sọ não, PET/CT não | ✅ **Sẵn sàng** — 4 loại u (glioma/meningioma/pituitary/no_tumor) |
| Gan | Siêu âm, CT, MRI, PET/CT | ❌ Chờ model 7 ung thư |
| Phổi | X-quang ngực, CT ngực, PET/CT | ❌ Chờ model 7 ung thư |
| Vú | Mammogram, siêu âm vú, MRI vú | ❌ Chờ model 7 ung thư |
| Dạ dày | Nội soi, CT, MRI, PET, EUS | ❌ Chờ model 7 ung thư |
| Đại trực tràng | Nội soi đại tràng, CT bụng-chậu, MRI trực tràng, PET | ❌ Chờ model 7 ung thư |
| Tuyến tiền liệt | MRI tuyến tiền liệt, siêu âm, PET/CT | ❌ Chờ model 7 ung thư |
| Cổ tử cung | MRI, CT, PET/CT | ❌ Chờ model 7 ung thư |

### Định dạng ảnh hỗ trợ

- **JPG / PNG**: ảnh thông thường
- **DICOM**: file `.dcm` và series DICOM
- **NIfTI**: volume `.nii` / `.nii.gz`
- **Pap/HPV, soi cổ tử cung, sinh thiết**: đầu vào lâm sàng, không hỗ trợ upload trực tiếp

Chat UI cho phép chọn nhóm bệnh và modality để lọc file picker phù hợp.

> **Lưu ý quan trọng**: Hệ thống hiện chỉ phân tích được **ung thư não** (model chính). Model 7 ung thư `medical_7_cancers_cnn.pt` chưa có — pipeline tự fallback sang brain model cho mọi ảnh. Khi upload ảnh, modality sẽ được tự động nhận diện (99.93% accuracy), nếu là não/MRI/CT sọ não → phân tích 4 loại u não; các loại khác báo "chỉ hỗ trợ não".

---

## 5. Dataset modality

Dataset `dataset/medical_modality/` dùng để train classifier phân loại modality ảnh y khoa:

- **Tổng: ~120.000 ảnh JPG** (8 modality × ~15.000/lớp), đã split thành `processed/images/{train,val,test}`
- 8 modality: CT, MRI, X-quang, Mammogram, Nội soi, Siêu âm, PET/CT, EUS
- Model: resnet18 @320px, batch 16, epochs 20, mixup 0.2, `num_workers=2`
- Train: `python run_medical.py train-modality`

---

## 6. Module chính trong `medical/`

| Module | Trách nhiệm |
|---|---|
| `dataset.py` | Tạo và kiểm tra layout dataset |
| `system_status.py` | Tổng hợp trạng thái medical |
| `training.py` | Audit, split, train, validate |
| `output_management.py` | Quản lý output medical |
| `storage.py` | Lưu và truy vấn case DB |
| `validator.py` | Kiểm tra ảnh đầu vào |
| `cli_helpers.py` | Helper in trạng thái CLI |

---

## 7. Hồ sơ cấu hình model (dùng khi tự train)

### CNN 7 ung thư / não (cấu hình cao — GPU 4GB VRAM)

Model 7 ung thư và não từng được train với cấu hình này (các script `run_train_*_high.py` đã gỡ khỏi repo, chỉ giữ hồ sơ tham số):

Model 7 ung thư: `medical_7_cancers_cnn.pt`; model não: `brain_classifier.pt` (đặt trong `models/pretrained/`).

| Tham số | 7 ung thư | Não |
|---|---|---|
| Backbone | convnext_tiny (pretrained) | convnext_tiny (pretrained) |
| Image size | 512px | 512px |
| Batch size | 4 (accum 4 → eff 16) | 4 (accum 4 → eff 16) |
| Epochs | 30 | 35 |
| Early-stop patience | 15 | 15 |
| Learning rate | 5e-5 | 3e-5 |
| Loss | Focal Loss γ=2 | Focal Loss γ=2 |
| Mixup | 0.2 | 0.2 |
| Label smoothing | 0.1 | 0.1 |
| EMA | 0.999 | 0.999 |
| Checkpoint averaging | Có (window 5) | Có (window 5) |
| Class weights | Có | Có |
| num_workers | 2 | 2 |
| Scheduler | cosine warmup restart | cosine warmup restart |
| fp16 + gradient clip 1.0 | Có | Có |
| Checkpoint mỗi 15 phút + resume | Có | Có |

**Thời gian ước tính:** não ~36 phút/epoch, 7 ung thư ~4.1h/epoch (đo trên RTX 3050 Ti, ảnh 512px, `num_workers=2`). Early-stop thường cắt ngắn đáng kể.

### CNN Modality (8 loại hình ảnh)

Script modality: `medical/modality_training.py` (giữ lại, model `modality_classifier.pt` đã train sẵn).

| Tham số | Giá trị |
|---|---|
| Backbone | resnet18 @320px |
| Batch size | 16 |
| Epochs | 20 |
| Learning rate | 1e-4 |
| Mixup | 0.2 |
| num_workers | 2 |

### Ngưỡng quyết định

| Ngưỡng | Giá trị | Mục đích |
|---|---|---|
| High risk | 0.35 | Ưu tiên recall, tránh bỏ sót |
| Medium risk | 0.25 | Cảnh báo sớm |
| Certainty | 0.30 | Threshold phân loại chung |

### YOLO Detection

| Tham số | Giá trị | Mục đích |
|---|---|---|
| Model | yolo11s | Nhẹ, phù hợp 4GB VRAM |
| Epochs | 50 | Train sâu |
| Imgsz | 512 | Chi tiết cao |
| Batch | 4 | An toàn VRAM |
| Optimizer | AdamW | Ổn định |
| Mosaic | 0.3 | Giảm nhiễu giải phẫu |
| Mixup | 0.1 | Tăng diversity |
| Copy-paste | 0.1 | Tăng diversity |
| Lật lên-xuống | 0.0 | Giữ hướng giải phẫu |

---

## 8. Debug theo triệu chứng

| Triệu chứng | Mở đầu tiên |
|---|---|
| `run_chat.py --check-only` fail | `utils/entrypoint_checks.py`, `medical/system_status.py` |
| Medical status sai | `medical/system_status.py`, `medical/model_policy.py` |
| Count train/val không đúng | `medical/training.py`, `medical/status_helpers.py` |
| Train medical fail | `medical/training.py`, `run_medical.py train` |
| CNN ảnh đầu vào lỗi | `medical/validator.py` |
