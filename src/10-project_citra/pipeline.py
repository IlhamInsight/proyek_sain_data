import hashlib
import json
import re
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

from feature_extraction import FEATURE_COLUMNS, extract_image_features


PROJECT_DIR = Path(__file__).resolve().parent
# Lokasi dataset berada di dalam folder 2-data_understanding/data
DATASET_DIR = PROJECT_DIR / "2-data_understanding" / "data"
PREPARATION_DIR = PROJECT_DIR / "3-data_preparation"
PROCESSED_DIR = PREPARATION_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "4-modeling" / "models"
EVALUATION_DIR = PROJECT_DIR / "5-evaluation" / "results"
DEPLOYMENT_DIR = PROJECT_DIR / "6-deployment"

CLASS_FOLDERS = {"RAPI": 1, "BERANTAKAN": 0}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TEST_GROUP = "kelompok 2"
LABEL_OVERRIDES = {
    "rapi/kelompok 1/berantakan 1.jpeg": (
        0,
        "Citra secara visual menunjukkan alas kaki berantakan.",
    ),
    "rapi/kelompok 1/berantakan 2.jpeg": (
        0,
        "Citra secara visual menunjukkan alas kaki berantakan.",
    ),
    "rapi/kelompok 1/whatsapp image 2026-11-08 at 17.44.57.jpeg": (
        0,
        "Duplikat citra berantakan yang juga tersimpan pada folder BERANTAKAN.",
    ),
    "rapi/kelompok 1/whatsapp image 2026-11-08 at 17.44.57_1.jpeg": (
        0,
        "Duplikat citra berantakan yang juga tersimpan pada folder BERANTAKAN.",
    ),
}


def _normalize_group(name: str) -> str:
    match = re.search(r"\d+", name)
    if not match:
        raise ValueError(f"Nama kelompok tidak memiliki nomor: {name}")
    return f"kelompok {match.group(0)}"


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    missing = set(FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Fitur tidak ditemukan pada dataset: {sorted(missing)}")
    return FEATURE_COLUMNS.copy()


def prepare_dataset() -> pd.DataFrame:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {DATASET_DIR}")

    records = []
    aug_flip_records = []
    label_review_candidates = []
    label_corrections = []
    seen_overrides = set()

    for folder_name, label in CLASS_FOLDERS.items():
        class_dir = DATASET_DIR / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_dir}")

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rel_path = image_path.relative_to(DATASET_DIR).as_posix()
            source_label = CLASS_FOLDERS[folder_name]
            effective_label = source_label
            override = LABEL_OVERRIDES.get(rel_path.casefold())
            if override is not None:
                effective_label, reason = override
                seen_overrides.add(rel_path.casefold())
                label_corrections.append(
                    {
                        "rel_path": rel_path,
                        "folder_label": "Rapi" if source_label == 1 else "Berantakan",
                        "corrected_label": (
                            "Rapi" if effective_label == 1 else "Berantakan"
                        ),
                        "reason": reason,
                    }
                )

            with Image.open(image_path) as image:
                features = extract_image_features(image)
                flipped = ImageOps.mirror(ImageOps.exif_transpose(image).convert("RGB"))
                features_flip = extract_image_features(flipped)

            filename_hint = image_path.stem.casefold()
            expected_hint = (
                0 if "berantakan" in filename_hint
                else 1 if "rapi" in filename_hint
                else None
            )
            if expected_hint is not None and effective_label != expected_hint:
                label_review_candidates.append(
                    {
                        "rel_path": rel_path,
                        "folder_label": (
                            "Rapi" if effective_label == 1 else "Berantakan"
                        ),
                        "filename_hint": "Berantakan" if expected_hint == 0 else "Rapi",
                        "review_reason": "Nama berkas mengindikasikan kelas yang berbeda dari label.",
                    }
                )

            kelompok_name = _normalize_group(image_path.parent.name)
            content_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            meta = {
                "filename": image_path.name,
                "rel_path": rel_path,
                "kelompok": kelompok_name,
                "class": "Rapi" if effective_label == 1 else "Berantakan",
                "label": effective_label,
                "folder_label": "Rapi" if source_label == 1 else "Berantakan",
                "label_corrected": source_label != effective_label,
                "include_in_model": True,
                "_content_hash": content_hash,
            }
            records.append({**features, **meta})
            aug_flip_records.append({**features_flip, **meta})

    frame = pd.DataFrame(records)
    frame_flip = pd.DataFrame(aug_flip_records)
    if frame.empty:
        raise ValueError("Dataset tidak memiliki citra yang dapat diproses.")
    if set(frame["label"]) != {0, 1}:
        raise ValueError("Dataset harus memuat citra dari kedua kelas.")
    if frame["rel_path"].duplicated().any():
        raise ValueError("Manifest dataset memiliki path gambar duplikat.")

    missing_overrides = set(LABEL_OVERRIDES).difference(seen_overrides)
    if missing_overrides:
        raise ValueError(
            f"Citra untuk koreksi label tidak ditemukan: {sorted(missing_overrides)}"
        )

    for _, duplicate_group in frame.groupby("_content_hash", sort=False):
        if len(duplicate_group) < 2:
            continue
        if duplicate_group["label"].nunique() > 1:
            paths = duplicate_group["rel_path"].tolist()
            raise ValueError(
                "Duplikat citra memiliki label yang bertentangan; periksa label "
                f"secara manual: {paths}"
            )
        canonical_path = duplicate_group.iloc[0]["rel_path"]
        duplicate_indexes = duplicate_group.index[1:]
        frame.loc[duplicate_indexes, "include_in_model"] = False
        frame_flip.loc[duplicate_indexes, "include_in_model"] = False
        for index in duplicate_indexes:
            label_review_candidates.append(
                {
                    "rel_path": frame.loc[index, "rel_path"],
                    "folder_label": frame.loc[index, "class"],
                    "filename_hint": "",
                    "review_reason": f"Duplikat identik; hanya satu salinan digunakan ({canonical_path}).",
                }
            )
    frame.drop(columns="_content_hash", inplace=True)
    frame_flip.drop(columns="_content_hash", inplace=True)

    groups = set(frame["kelompok"])
    if TEST_GROUP not in groups:
        raise ValueError(
            f"Kelompok uji harus tersedia ({TEST_GROUP}); kelompok ditemukan: {sorted(groups)}"
        )

    frame["split"] = "train"
    frame.loc[frame["kelompok"] == TEST_GROUP, "split"] = "test"
    frame_flip["split"] = frame["split"]

    split_groups = {
        split: set(part["kelompok"])
        for split, part in frame.groupby("split")
    }
    if split_groups["train"] & split_groups["test"]:
        raise ValueError("Ditemukan kelompok yang bocor lintas split.")
    for split, part in frame.groupby("split"):
        if set(part["label"]) != {0, 1}:
            raise ValueError(f"Split {split} tidak memuat kedua kelas.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PREPARATION_DIR / "dataset_manifest.csv"
    frame.to_csv(PROCESSED_DIR / "dataset_manifest.csv", index=False)
    frame.to_csv(manifest_path, index=False)

    frame_flip.loc[
        (frame_flip["split"] == "train") & frame_flip["include_in_model"]
    ].to_csv(PROCESSED_DIR / "train_flip_features.csv", index=False)

    review_path = PROCESSED_DIR / "label_review_candidates.csv"
    pd.DataFrame(
        label_review_candidates,
        columns=["rel_path", "folder_label", "filename_hint", "review_reason"],
    ).to_csv(review_path, index=False)
    pd.DataFrame(
        label_corrections,
        columns=["rel_path", "folder_label", "corrected_label", "reason"],
    ).to_csv(PROCESSED_DIR / "label_corrections.csv", index=False)

    for split in ("train", "test"):
        frame.loc[
            (frame["split"] == split) & frame["include_in_model"]
        ].to_csv(
            PROCESSED_DIR / f"{split}_features.csv", index=False
        )

    feature_columns = _feature_columns(frame)

    print(
        f"Dataset: {len(frame)} citra, "
        f"{int(frame['include_in_model'].sum())} citra unik dipakai; "
        f"fitur pemodelan: {len(feature_columns)}"
    )
    print(
        f"Koreksi label: {int(frame['label_corrected'].sum())}; "
        f"duplikat dikecualikan: {int((~frame['include_in_model']).sum())}"
    )
    print(
        frame.loc[frame["include_in_model"]]
        .groupby(["split", "class"])
        .size()
        .unstack(fill_value=0)
    )
    print(f"Train: semua kelompok kecuali {TEST_GROUP}; Test (Unseen): {TEST_GROUP}")
    print(f"Manifest tersimpan: {manifest_path}")
    return frame


def _build_candidate_models() -> dict:
    """
    Kandidat model terregularisasi untuk mencegah overfitting pada 120 citra pengembangan
    sekaligus memaksimalkan kemampuan generalisasi pada kelompok baru.
    """
    return {
        "LogReg_PCA10": make_pipeline(
            PCA(n_components=10, random_state=42),
            LogisticRegression(
                C=0.1, max_iter=2000, class_weight="balanced", random_state=42
            ),
        ),
        "LogReg_PCA8": make_pipeline(
            PCA(n_components=8, random_state=42),
            LogisticRegression(
                C=0.1, max_iter=2000, class_weight="balanced", random_state=42
            ),
        ),
        "LogReg_KBest25": make_pipeline(
            SelectKBest(f_classif, k=25),
            LogisticRegression(
                C=0.15, max_iter=2000, class_weight="balanced", random_state=42
            ),
        ),
        "SVC_PCA10": make_pipeline(
            PCA(n_components=10, random_state=42),
            SVC(
                kernel="rbf",
                C=0.5,
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "SVC_KBest20": make_pipeline(
            SelectKBest(f_classif, k=20),
            SVC(
                kernel="rbf",
                C=0.3,
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "MLP_Regularized": make_pipeline(
            SelectKBest(f_classif, k=20),
            MLPClassifier(
                hidden_layer_sizes=(16, 8),
                alpha=4.0,
                max_iter=2000,
                random_state=42,
            ),
        ),
        "LGBM_Regularized": make_pipeline(
            SelectKBest(f_classif, k=25),
            LGBMClassifier(
                n_estimators=40,
                max_depth=2,
                num_leaves=4,
                min_child_samples=20,
                reg_alpha=1.5,
                reg_lambda=10.0,
                colsample_bytree=0.5,
                subsample=0.8,
                subsample_freq=1,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            ),
        ),
        "RF_Regularized": make_pipeline(
            SelectKBest(f_classif, k=25),
            RandomForestClassifier(
                n_estimators=200,
                max_depth=3,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    }


def train_model(frame: pd.DataFrame | None = None) -> dict:
    manifest_path = PROCESSED_DIR / "dataset_manifest.csv"
    flip_path = PROCESSED_DIR / "train_flip_features.csv"
    if frame is None:
        if not manifest_path.is_file() or not flip_path.is_file():
            frame = prepare_dataset()
        else:
            frame = pd.read_csv(manifest_path)
            if "include_in_model" not in frame or "label_corrected" not in frame:
                frame = prepare_dataset()
    elif not flip_path.is_file():
        frame = prepare_dataset()

    feature_columns = _feature_columns(frame)
    eligible = frame["include_in_model"] if "include_in_model" in frame else True
    training_frame = frame.loc[
        (frame["split"] != "test") & eligible
    ].reset_index(drop=True)
    if training_frame.empty or set(training_frame["label"]) != {0, 1}:
        raise ValueError("Data pengembangan kosong atau tidak memuat kedua kelas.")
    if training_frame["kelompok"].nunique() < 3:
        raise ValueError("Validasi group-aware membutuhkan setidaknya tiga kelompok train.")

    flip_frame = pd.read_csv(flip_path)
    flip_aligned = training_frame[["rel_path"]].merge(flip_frame, on="rel_path", how="left")
    if flip_aligned[feature_columns].isna().any().any():
        raise ValueError("Data augmentasi horizontal tidak lengkap untuk subset latih.")

    X = training_frame[feature_columns].to_numpy()
    X_flip = flip_aligned[feature_columns].to_numpy()
    y = training_frame["label"].to_numpy()
    groups = training_frame["kelompok"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(X, y, groups))
    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        if set(y[train_index]) != {0, 1} or set(y[validation_index]) != {0, 1}:
            raise ValueError(
                f"Fold {fold_number} tidak memuat kedua kelas pada train dan validasi."
            )
        if set(groups[train_index]) & set(groups[validation_index]):
            raise ValueError(f"Kelompok bocor pada fold {fold_number}.")

    models = _build_candidate_models()
    results = []
    best_name = None
    best_model = None
    best_macro_f1 = -1.0

    for name, candidate in models.items():
        fold_tr_acc, fold_acc, fold_f1, fold_prec, fold_rec = [], [], [], [], []
        for train_index, validation_index in folds:
            X_tr = np.vstack([X[train_index], X_flip[train_index]])
            y_tr = np.concatenate([y[train_index], y[train_index]])
            X_val = X[validation_index]
            y_val = y[validation_index]

            # PowerTransformer (Yeo-Johnson) di-fit murni di dalam train fold
            scaler_fold = PowerTransformer(method="yeo-johnson", standardize=True).fit(X_tr)
            X_tr_scaled = scaler_fold.transform(X_tr)
            X_val_scaled = scaler_fold.transform(X_val)

            est = clone(candidate)
            est.fit(X_tr_scaled, y_tr)

            preds_tr = est.predict(X_tr_scaled)
            preds_val = est.predict(X_val_scaled)

            fold_tr_acc.append(accuracy_score(y_tr, preds_tr))
            fold_acc.append(accuracy_score(y_val, preds_val))
            fold_f1.append(f1_score(y_val, preds_val, average="macro", zero_division=0))
            fold_prec.append(precision_score(y_val, preds_val, average="macro", zero_division=0))
            fold_rec.append(recall_score(y_val, preds_val, average="macro", zero_division=0))

        tr_acc_mean = float(np.mean(fold_tr_acc))
        val_acc_mean = float(np.mean(fold_acc))
        macro_f1 = float(np.mean(fold_f1))
        result = {
            "model": name,
            "train_accuracy_mean": tr_acc_mean,
            "validation_accuracy_mean": val_acc_mean,
            "validation_accuracy_std": float(np.std(fold_acc)),
            "train_val_gap": float(tr_acc_mean - val_acc_mean),
            "validation_macro_f1": macro_f1,
            "validation_macro_f1_std": float(np.std(fold_f1)),
            "validation_precision_macro_mean": float(np.mean(fold_prec)),
            "validation_recall_macro_mean": float(np.mean(fold_rec)),
        }
        results.append(result)
        if macro_f1 > best_macro_f1:
            best_name = name
            best_model = candidate
            best_macro_f1 = macro_f1

    if best_model is None or best_name is None:
        raise RuntimeError("Tidak ada model yang berhasil dilatih.")

    X_dev_aug = np.vstack([X, X_flip])
    y_dev_aug = np.concatenate([y, y])

    final_scaler = PowerTransformer(method="yeo-johnson", standardize=True).fit(X_dev_aug)
    X_training = final_scaler.transform(X_dev_aug)
    final_model = clone(best_model)
    final_model.fit(X_training, y_dev_aug)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_DIR / "feature_extraction.py", DEPLOYMENT_DIR / "feature_extraction.py")
    with (MODEL_DIR / "feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)
    with (MODEL_DIR / "model_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    joblib.dump(final_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(final_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(final_scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(final_model, DEPLOYMENT_DIR / "best_model.joblib")
    joblib.dump(final_scaler, DEPLOYMENT_DIR / "scaler.joblib")
    with (DEPLOYMENT_DIR / "feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)

    train_acc_final = accuracy_score(y_dev_aug, final_model.predict(X_training))
    print(f"Model terpilih: {best_name} (macro-F1 validasi {best_macro_f1:.3f}, train_acc {train_acc_final:.3f})")
    print(f"Model akhir dilatih pada {len(training_frame)} citra unik (+{len(training_frame)} augmentasi horizontal = {len(X_dev_aug)} sampel).")
    return {"model_name": best_name, "model": final_model, "results": results}


def evaluate_model() -> dict:
    train_path = PROCESSED_DIR / "train_features.csv"
    test_path = PROCESSED_DIR / "test_features.csv"
    model_path = MODEL_DIR / "best_model.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    feature_path = MODEL_DIR / "feature_columns.json"
    if not all(path.is_file() for path in (train_path, test_path, model_path, scaler_path, feature_path)):
        raise FileNotFoundError("Artefak test/model belum lengkap; jalankan prepare dan train.")

    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    with feature_path.open(encoding="utf-8") as file:
        feature_columns = json.load(file)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if set(test_frame["split"]) != {"test"}:
        raise ValueError("Berkas test_features.csv berisi data selain split test.")
    test_groups = set(test_frame["kelompok"])
    train_val = pd.read_csv(PROCESSED_DIR / "dataset_manifest.csv")
    train_val_groups = set(
        train_val.loc[train_val["split"].isin(["train", "val"]), "kelompok"]
    )
    if test_groups & train_val_groups:
        raise ValueError("Kelompok pada test set juga muncul pada train/validasi.")

    X_train_scaled = scaler.transform(train_frame[feature_columns].to_numpy())
    y_train = train_frame["label"].to_numpy()
    train_preds = model.predict(X_train_scaled)
    train_acc = float(accuracy_score(y_train, train_preds))
    train_f1_macro = float(f1_score(y_train, train_preds, average="macro", zero_division=0))

    X_test = scaler.transform(test_frame[feature_columns].to_numpy())
    y_test = test_frame["label"].to_numpy()
    predictions = model.predict(X_test)
    probability_by_class = {
        int(label): probability
        for label, probability in zip(model.classes_, model.predict_proba(X_test).T)
    }
    positive_probabilities = probability_by_class[1]

    test_acc = float(accuracy_score(y_test, predictions))
    test_f1_macro = float(f1_score(y_test, predictions, average="macro", zero_division=0))

    metrics = {
        "model_name": type(model).__name__,
        "test_groups": sorted(test_groups),
        "train_samples": int(len(train_frame)),
        "train_accuracy": train_acc,
        "train_f1_macro": train_f1_macro,
        "test_samples": int(len(test_frame)),
        "test_accuracy": test_acc,
        "train_test_gap": float(train_acc - test_acc),
        "test_precision_weighted": float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "test_recall_weighted": float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "test_f1_weighted": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "test_f1_macro": test_f1_macro,
        "test_roc_auc": float(roc_auc_score(y_test, positive_probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Berantakan", "Rapi"],
            output_dict=True,
            zero_division=0,
        ),
        "target_accuracy_met": bool(test_acc >= 0.85),
        "target_weighted_f1_met": bool(
            f1_score(y_test, predictions, average="weighted", zero_division=0) >= 0.83
        ),
        "target_macro_f1_met": bool(test_f1_macro >= 0.83),
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    with (EVALUATION_DIR / "evaluation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    pd.DataFrame(
        {
            "rel_path": test_frame["rel_path"],
            "kelompok": test_frame["kelompok"],
            "actual_label": y_test,
            "predicted_label": predictions,
            "probability_rapi": positive_probabilities,
        }
    ).to_csv(EVALUATION_DIR / "test_predictions.csv", index=False)

    print(
        f"Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f} (Gap={train_acc - test_acc:+.3f}), "
        f"Test Macro-F1={test_f1_macro:.3f}, ROC-AUC={metrics['test_roc_auc']:.3f}"
    )
    return metrics


def run_pipeline() -> dict:
    frame = prepare_dataset()
    training = train_model(frame)
    metrics = evaluate_model()
    return {"training": training, "evaluation": metrics}
