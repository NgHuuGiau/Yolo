from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from medical.cancer_catalog import (
    COMMON_CANCER_TARGETS,
    supported_cancer_labels,
    supported_cancer_modalities,
)
from medical.dataset import MEDICAL_CLASS_NAMES
from medical.model_policy import resolve_medical_runtime_model_path
from medical.pipeline import build_default_medical_analyzer_config
from medical.status_helpers import count_files, count_medical_images
from medical.training import medical_training_paths

SCREENING_TARGETS = tuple((target.label, target.model_ready) for target in COMMON_CANCER_TARGETS)
ANALYZED_CANCERS = tuple(supported_cancer_labels())


@dataclass(frozen=True)
class MedicalSystemStatus:
    configured_model_path: Path
    resolved_model_path: Path | None
    allow_fallback_model: bool
    using_fallback_model: bool
    model_ready: bool
    model_message: str
    dataset_root: Path
    data_yaml_path: Path
    raw_images: int
    raw_labels: int
    train_images: int
    val_images: int
    test_images: int
    report_files: int
    normalized_files: int
    overlay_files: int
    export_files: int
    case_db_path: Path
    case_count: int
    screening_targets: tuple[tuple[str, bool], ...]
    analyzed_cancers: tuple[str, ...]
    analyzed_modalities: tuple[str, ...]
    total_images: int = 0

    @property
    def dataset_initialized(self) -> bool:
        return self.data_yaml_path.exists()

    @property
    def raw_dataset_ready(self) -> bool:
        return self.total_images > 0

    @property
    def processed_dataset_ready(self) -> bool:
        return self.train_images > 0 and self.val_images > 0 and self.test_images > 0


def _count_cases(case_db_path: Path) -> int:
    if not case_db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(case_db_path)
        try:
            row = conn.execute("SELECT COUNT(*) FROM medical_cases").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return 0


def _count_split_images(dataset_root: Path, split: str) -> int:
    return sum(
        count_medical_images(dataset_root / class_name / "processed" / "images" / split)
        for class_name in MEDICAL_CLASS_NAMES
    )


def get_medical_system_status() -> MedicalSystemStatus:
    config = build_default_medical_analyzer_config()
    training_paths = medical_training_paths()
    report_dir = Path(config.working_dir) / "reports"
    normalized_dir = Path(config.working_dir) / "normalized_images"
    overlay_dir = Path(config.working_dir) / "processed_images"
    export_dir = Path(config.working_dir) / "exports"
    case_db_path = Path(config.working_dir).parent / "onco.db"

    try:
        resolved_model_path = resolve_medical_runtime_model_path(config)
        resolved_model_abs = resolved_model_path.resolve(strict=False)
        configured_model_abs = config.model_path.resolve(strict=False)
        bundled_model_abs = (Path("medical") / Path(config.model_path).name).resolve(strict=False)
        fallback_model_abs = (
            Path(config.fallback_model_path).resolve(strict=False)
            if config.allow_fallback_model and config.fallback_model_path is not None
            else None
        )
        using_fallback_model = fallback_model_abs is not None and resolved_model_abs == fallback_model_abs
        model_ready = True
        if resolved_model_abs == configured_model_abs:
            model_message = f"Da san sang voi model: {resolved_model_path}"
        elif using_fallback_model:
            model_message = f"Dang dung fallback model: {resolved_model_path}"
        elif resolved_model_abs == bundled_model_abs:
            model_message = f"Dang dung model dong goi trong medical/: {resolved_model_path}"
        else:
            model_message = f"Dang dung model medical tai: {resolved_model_path}"
    except Exception as exc:
        resolved_model_path = None
        using_fallback_model = False
        brain_model_path = Path(config.brain_model_path) if config.brain_model_path else None
        if brain_model_path is not None and brain_model_path.exists():
            resolved_model_path = brain_model_path
            model_ready = True
            model_message = (
                f"Chua co model tong quat, dang dung brain model ({brain_model_path.name}). "
                f"Chỉ phân tích được ảnh vùng đầu (body_region=brain)."
            )
        else:
            model_ready = False
            model_message = str(exc)

    train_images = _count_split_images(training_paths.dataset_root, "train")
    val_images = _count_split_images(training_paths.dataset_root, "val")
    test_images = _count_split_images(training_paths.dataset_root, "test")

    return MedicalSystemStatus(
        configured_model_path=config.model_path,
        resolved_model_path=resolved_model_path,
        allow_fallback_model=config.allow_fallback_model,
        using_fallback_model=using_fallback_model,
        model_ready=model_ready,
        model_message=model_message,
        dataset_root=training_paths.dataset_root,
        data_yaml_path=training_paths.data_yaml_path,
        raw_images=train_images + val_images + test_images,
        raw_labels=train_images + val_images + test_images,
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
        total_images=train_images + val_images + test_images,
        report_files=count_files(report_dir),
        normalized_files=count_files(normalized_dir),
        overlay_files=count_files(overlay_dir),
        export_files=count_files(export_dir),
        case_db_path=case_db_path,
        case_count=_count_cases(case_db_path),
        screening_targets=SCREENING_TARGETS,
        analyzed_cancers=ANALYZED_CANCERS,
        analyzed_modalities=tuple(supported_cancer_modalities()),
    )


def recommended_medical_commands(status: MedicalSystemStatus) -> list[str]:
    commands: list[str] = []
    if not status.dataset_initialized:
        commands.append("python run_medical.py init-dataset")
    if status.raw_dataset_ready and not status.processed_dataset_ready:
        commands.append("python run_medical.py split-dataset")
    elif not status.raw_dataset_ready:
        commands.append("python run_medical.py audit-dataset")
    if not status.model_ready:
        commands.append("Bổ sung file model đã train vào models/pretrained/ (vd: medical_7_cancers_cnn.pt)")
    else:
        commands.append("python run_medical.py validate")
    if any((status.report_files, status.normalized_files, status.overlay_files, status.export_files)):
        commands.append("python run_chat.py --cleanup-output --older-than-days 30")
    return list(dict.fromkeys(commands))
