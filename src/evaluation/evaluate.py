"""Model Evaluation Pipeline.

Loads best model, runs inference on test dataset, computes metrics
(Accuracy, F1, Balanced Accuracy, Cohen's Kappa, MCC, ECE, MCE),
generates diagrams (confusion matrices, ROC, Precision-Recall, Reliability),
performs error audits, builds Grad-CAM galleries, and exports HTML, MD, and JSON reports.
"""

import os
import cv2  # HIGH-11: was missing — required for imread/resize throughout pipeline
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    classification_report
)

from src import config
from src.utils import set_seed
from src.data.dataset_loader import MRIDatasetLoader
from src.data.augmentation import MRIAugmentationPipeline, RandomShear  # CRITICAL-05: needed for custom_objects
from src.models.efficientnet_model import load_model_robustly
from src.evaluation.explainability.gradcam import GradCAMExplainer

# Setup custom logging for evaluation
log_path = Path(config.LOG_DIR)
log_path.mkdir(parents=True, exist_ok=True)
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
eval_log_handler = logging.FileHandler(log_path / "evaluation.log", encoding="utf-8")
eval_log_handler.setFormatter(formatter)

logger = logging.getLogger("EvaluationPipeline")
logger.setLevel(logging.INFO)
logger.addHandler(eval_log_handler)
# Direct to stdout as well
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def get_system_metadata() -> Dict[str, Any]:
    """Retrieves hardware, execution environment, and package versions metadata.

    Returns:
        Dict[str, Any]: Metadata dictionary.
    """
    gpus = tf.config.list_physical_devices("GPU")
    device_used = "GPU" if gpus else "CPU"
    
    return {
        "tensorflow_version": tf.__version__,
        "device_used": device_used,
        "gpu_count": len(gpus),
        "model_version": config.VERSION,
        "random_seed": config.SEED,
    }

def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10
) -> Tuple[float, float, List[float], List[float], List[int]]:
    """Calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        y_true (np.ndarray): True labels indices.
        y_prob (np.ndarray): Predicted probability vectors of shape (N, C).
        num_bins (int): Number of bins to group confidences. Defaults to 10.

    Returns:
        Tuple[float, float, List[float], List[float], List[int]]:
            ECE, MCE, average accuracy per bin, average confidence per bin, and size of each bin.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    ece = 0.0
    mce = 0.0
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        # Select items in this confidence interval
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = int(np.sum(in_bin))
        bin_sizes.append(bin_size)
        
        prop_in_bin = bin_size / len(y_true)
        
        if bin_size > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            
            bin_accs.append(float(accuracy_in_bin))
            bin_confs.append(float(confidence_in_bin))
            
            difference = np.abs(accuracy_in_bin - confidence_in_bin)
            ece += prop_in_bin * difference
            mce = max(mce, difference)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            
    return float(ece), float(mce), bin_accs, bin_confs, bin_sizes

def plot_reliability_diagram(
    bin_accs: List[float],
    bin_confs: List[float],
    ece: float,
    mce: float,
    save_path: Path
) -> None:
    """Generates and saves the reliability calibration diagram.

    Args:
        bin_accs (List[float]): Binned accuracy rates.
        bin_confs (List[float]): Binned average confidences.
        ece (float): Expected Calibration Error.
        mce (float): Maximum Calibration Error.
        save_path (Path): File path to save the diagram.
    """
    plt.figure(figsize=(7, 7))
    num_bins = len(bin_accs)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    
    # Plot bars
    plt.bar(
        bin_centers,
        bin_accs,
        width=1.0 / num_bins,
        edgecolor="black",
        color="royalblue",
        alpha=0.8,
        label="Model Outputs"
    )
    
    # Diagonal baseline reference (perfect calibration)
    plt.plot([0, 1], [0, 1], "r--", label="Perfect Calibration")
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram (ECE={ece:.4f} | MCE={mce:.4f})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confidence_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str],
    save_path: Path
) -> None:
    """Generates histograms analyzing confidence intervals for predictions.

    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Probabilities vector.
        classes (List[str]): List of class names.
        save_path (Path): Target save path.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Correct vs Incorrect prediction confidences
    sns.histplot(
        confidences[accuracies],
        bins=10,
        color="green",
        label="Correct Predictions",
        alpha=0.6,
        ax=axes[0],
        kde=True,
        element="step"
    )
    sns.histplot(
        confidences[~accuracies],
        bins=10,
        color="red",
        label="Incorrect Predictions",
        alpha=0.6,
        ax=axes[0],
        kde=True,
        element="step"
    )
    axes[0].set_title("Prediction Confidence Distribution (Correct vs Incorrect)")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Plot 2: Average confidence per class
    avg_confs = []
    for i, cls in enumerate(classes):
        cls_mask = (y_true == i)
        if np.any(cls_mask):
            avg_confs.append(np.mean(confidences[cls_mask]))
        else:
            avg_confs.append(0.0)
            
    sns.barplot(
        x=classes,
        y=avg_confs,
        ax=axes[1],
        palette="viridis",
        edgecolor="black"
    )
    axes[1].set_title("Average Prediction Confidence per Class")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Average Confidence")
    axes[1].set_ylim(0, 1)
    for i, val in enumerate(avg_confs):
        axes[1].text(i, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    save_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates, saves, and exports confusion matrices (Raw & Normalized).

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        classes (List[str]): List of class names.
        save_dir (Path): Output directory.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Raw CM DataFrame, Normalized CM DataFrame.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    
    # 1. Raw CM Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        linewidths=0.5,
        edgecolor="gray"
    )
    plt.title("Confusion Matrix (Raw Counts)")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png")
    plt.close()
    
    # 2. Normalized CM Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        linewidths=0.5,
        edgecolor="gray"
    )
    plt.title("Confusion Matrix (Normalized Rates)")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(save_dir / "normalized_confusion_matrix.png")
    plt.close()
    
    # Convert to DataFrames and export to CSV
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_norm_df = pd.DataFrame(cm_norm, index=classes, columns=classes)
    
    cm_df.to_csv(save_dir / "confusion_matrix.csv")
    cm_norm_df.to_csv(save_dir / "normalized_confusion_matrix.csv")
    
    return cm_df, cm_norm_df

def plot_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str],
    save_dir: Path
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Plots and saves multi-class One-vs-Rest ROC and Precision-Recall Curves.

    Args:
        y_true (np.ndarray): True class labels indices.
        y_prob (np.ndarray): Model outputs probabilities matrix.
        classes (List[str]): List of diagnosis classes.
        save_dir (Path): Target folder.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: Dict of class-wise ROC AUC values,
                                                 and Class-wise PR AUC values.
    """
    n_classes = len(classes)
    
    # One-hot encode ground truth
    y_true_onehot = np.eye(n_classes)[y_true]
    
    roc_aucs = {}
    pr_aucs = {}
    
    # --- 1. ROC CURVE ---
    plt.figure(figsize=(8, 7))
    
    # Plot class curves
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        roc_aucs[classes[i]] = float(roc_auc)
        plt.plot(fpr, tpr, label=f"{classes[i]} (AUC = {roc_auc:.3f})")
        
    # Micro Average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), y_prob.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(
        fpr_micro,
        tpr_micro,
        label=f"Micro-Average (AUC = {roc_auc_micro:.3f})",
        linestyle=":",
        color="deeppink",
        linewidth=3
    )
    
    # Macro Average ROC
    all_fpr = np.unique(np.concatenate([roc_curve(y_true_onehot[:, i], y_prob[:, i])[0] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, roc_curve(y_true_onehot[:, i], y_prob[:, i])[0], roc_curve(y_true_onehot[:, i], y_prob[:, i])[1])
    mean_tpr /= n_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)
    plt.plot(
        all_fpr,
        mean_tpr,
        label=f"Macro-Average (AUC = {roc_auc_macro:.3f})",
        linestyle=":",
        color="navy",
        linewidth=3
    )
    
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest (OvR) Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curve.png")
    plt.close()
    
    # --- 2. PRECISION-RECALL CURVE ---
    plt.figure(figsize=(8, 7))
    
    # Plot class curves
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_prob[:, i])
        pr_auc = auc(recall, precision)
        pr_aucs[classes[i]] = float(pr_auc)
        plt.plot(recall, precision, label=f"{classes[i]} (PR-AUC = {pr_auc:.3f})")
        
    # Micro Average PR
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_onehot.ravel(), y_prob.ravel())
    pr_auc_micro = auc(recall_micro, precision_micro)
    plt.plot(
        recall_micro,
        precision_micro,
        label=f"Micro-Average (PR-AUC = {pr_auc_micro:.3f})",
        linestyle=":",
        color="deeppink",
        linewidth=3
    )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("One-vs-Rest (OvR) Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "pr_curve.png")
    plt.close()
    
    return roc_aucs, pr_aucs

def generate_gradcam_gallery(
    model: tf.keras.Model,
    df: pd.DataFrame,
    classes: List[str],
    output_dir: Path
) -> List[Dict[str, Any]]:
    """Generates a Grad-CAM gallery of correctly and incorrectly predicted MRI scans.

    Saves overlaid heatmaps and builds index mappings.

    Args:
        model (tf.keras.Model): Trained Keras model.
        df (pd.DataFrame): Prediction outputs DataFrame.
        classes (List[str]): List of classes.
        output_dir (Path): Output visualisations folder.

    Returns:
        List[Dict[str, Any]]: List of dictionary reports containing Grad-CAM path metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    explainer = GradCAMExplainer()
    gallery_info = []

    # Map class to index
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Target: At least 5 correct and 5 incorrect predictions per class (if available)
    for class_name in classes:
        class_idx = class_to_idx[class_name]
        class_df = df[df["true_class"] == class_name]
        
        # 1. Select Correct Predictions
        correct_df = class_df[class_df["is_correct"] == True]
        correct_samples = correct_df.sort_values(by="confidence", ascending=False).head(5)
        
        # 2. Select Incorrect Predictions
        incorrect_df = class_df[class_df["is_correct"] == False]
        incorrect_samples = incorrect_df.sort_values(by="confidence", ascending=False).head(5)

        for mode, samples in [("correct", correct_samples), ("incorrect", incorrect_samples)]:
            for idx, (_, row) in enumerate(samples.iterrows()):
                try:
                    img_path = Path(row["file_path"])
                    if not img_path.exists():
                        continue
                        
                    # Load grayscale image
                    img_raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img_raw is None:
                        continue
                        
                    # Resize to (width, height) — cv2.resize expects (w, h) order (HIGH-08)
                    img_resized = cv2.resize(img_raw, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
                    img_tensor = tf.cast(img_resized, tf.float32) / 255.0
                    img_tensor = tf.expand_dims(img_tensor, axis=-1)
                    img_tensor = tf.image.grayscale_to_rgb(img_tensor)  # 3-channels
                    
                    # Target category for explanation
                    target_idx = class_to_idx[row["predicted_class"]]
                    
                    # Generate Heatmap
                    heatmap = explainer.explain(model, img_tensor, target_idx)
                    
                    # Overlay heatmap on original BGR/gray image
                    overlaid = explainer.overlay_heatmap(img_resized, heatmap)
                    
                    # Save overlaid image
                    filename = f"gradcam_{class_name}_{mode}_{idx}.png"
                    save_path = output_dir / filename
                    # Save as BGR for OpenCV
                    cv2.imwrite(str(save_path), cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
                    
                    gallery_info.append({
                        "filename": filename,
                        "relative_path": f"gradcam/{filename}",
                        "original_path": row["file_path"],
                        "true_class": row["true_class"],
                        "predicted_class": row["predicted_class"],
                        "confidence": float(row["confidence"]),
                        "mode": mode
                    })
                except Exception as e:
                    logger.warning(f"Failed to generate Grad-CAM for {row['filename']}: {e}")
                    
    logger.info(f"Generated {len(gallery_info)} Grad-CAM overlay visualizations.")
    return gallery_info

def generate_html_report(
    summary: Dict[str, Any],
    misclassification_data: Dict[str, List[Dict[str, Any]]],
    gradcam_gallery: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """Generates a professional, responsive HTML evaluation report with navigation sidebar.

    Args:
        summary (Dict[str, Any]): Overall stats and configurations.
        misclassification_data (Dict[str, List[Dict[str, Any]]]): Wrong and Borderline corrects.
        gradcam_gallery (List[Dict[str, Any]]): Grad-CAM image references.
        output_path (Path): Path to output HTML.
    """
    
    def get_color_class(val: float) -> str:
        if val >= 0.85:
            return "badge-success"
        elif val >= 0.70:
            return "badge-warning"
        return "badge-danger"

    # Compile HTML text
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Report - Brain MRI Classification</title>
    <style>
        :root {{
            --primary: #2C3E50;
            --secondary: #18BC9C;
            --dark: #343A40;
            --light: #F8F9FA;
            --success: #28A745;
            --warning: #FFC107;
            --danger: #DC3545;
            --border: #DEE2E6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #F4F6F9;
            color: #333;
            display: flex;
            min-height: 100vh;
        }}
        /* Sidebar Navigation */
        .sidebar {{
            width: 260px;
            background-color: var(--primary);
            color: white;
            padding: 20px;
            position: fixed;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        .sidebar h2 {{
            font-size: 1.2rem;
            margin-bottom: 25px;
            text-align: center;
            border-bottom: 2px solid rgba(255,255,255,0.1);
            padding-bottom: 10px;
        }}
        .sidebar a {{
            color: rgba(255,255,255,0.8);
            text-decoration: none;
            padding: 12px 15px;
            margin-bottom: 5px;
            border-radius: 4px;
            font-size: 0.95rem;
            transition: all 0.2s;
        }}
        .sidebar a:hover, .sidebar a.active {{
            background-color: var(--secondary);
            color: white;
            font-weight: bold;
        }}
        /* Main Layout */
        .main-content {{
            margin-left: 260px;
            padding: 40px;
            flex: 1;
            max-width: 1200px;
        }}
        header {{
            margin-bottom: 30px;
            border-bottom: 3px solid var(--secondary);
            padding-bottom: 15px;
        }}
        header h1 {{ color: var(--primary); font-size: 2.2rem; margin-bottom: 5px; }}
        header p {{ color: #7F8C8D; }}
        
        .section {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid var(--border);
        }}
        .section h2 {{
            color: var(--primary);
            margin-bottom: 20px;
            font-size: 1.5rem;
            border-left: 5px solid var(--secondary);
            padding-left: 10px;
        }}
        /* Summary Cards */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 6px;
            padding: 20px;
            text-align: center;
            border: 1px solid var(--border);
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transition: transform 0.2s;
        }}
        .card:hover {{ transform: translateY(-3px); }}
        .card h3 {{ font-size: 0.9rem; color: #7F8C8D; margin-bottom: 10px; text-transform: uppercase; }}
        .card .value {{ font-size: 1.8rem; font-weight: bold; color: var(--primary); }}
        
        .badge-success {{ color: var(--success); }}
        .badge-warning {{ color: var(--warning); }}
        .badge-danger {{ color: var(--danger); }}
        
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        table th, table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        table th {{ background-color: var(--light); color: var(--primary); font-weight: 600; }}
        table tr:hover {{ background-color: #F8F9FA; }}
        
        /* Dashboard Flex grids */
        .image-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 15px;
        }}
        .image-col {{
            flex: 1;
            min-width: 300px;
            background: white;
            border-radius: 6px;
            border: 1px solid var(--border);
            padding: 15px;
            text-align: center;
        }}
        .image-col img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid var(--border);
        }}
        .image-col p {{ font-size: 0.9rem; color: #7F8C8D; margin-top: 10px; }}
        
        /* Gradcam gallery grid */
        .gradcam-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .gradcam-card {{
            background: var(--light);
            border-radius: 6px;
            padding: 10px;
            border: 1px solid var(--border);
            text-align: center;
        }}
        .gradcam-card img {{
            width: 100%;
            border-radius: 4px;
            border: 1px solid var(--border);
        }}
        .gradcam-card .title {{ font-size: 0.8rem; font-weight: bold; margin-top: 8px; }}
        .gradcam-card .info {{ font-size: 0.75rem; color: #7F8C8D; margin-top: 4px; }}
        
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>MRI CLASSIFIER</h2>
        <a href="#summary" class="active">Overview Summary</a>
        <a href="#metrics">Detailed Metrics</a>
        <a href="#confusion">Confusion Matrix</a>
        <a href="#curves">ROC & PR Curves</a>
        <a href="#calibration">Calibration Info</a>
        <a href="#gradcam">Grad-CAM Visuals</a>
        <a href="#misclassifications">Failure Audits</a>
        <a href="#conclusions">Conclusions</a>
    </div>
    
    <div class="main-content">
        <header>
            <h1>Evaluation & Diagnostics Report</h1>
            <p>Project: <strong>Brain MRI Tumor Classification</strong> | Model Architecture: <strong>EfficientNetB0</strong></p>
        </header>
        
        <!-- SECTION 1: OVERVIEW -->
        <div id="summary" class="section">
            <h2>Overview Summary</h2>
            <div class="metrics-grid">
                <div class="card">
                    <h3>Accuracy</h3>
                    <div class="value {get_color_class(summary['accuracy'])}">{summary['accuracy']:.2%}</div>
                </div>
                <div class="card">
                    <h3>Balanced Acc</h3>
                    <div class="value {get_color_class(summary['balanced_accuracy'])}">{summary['balanced_accuracy']:.2%}</div>
                </div>
                <div class="card">
                    <h3>Overall F1</h3>
                    <div class="value {get_color_class(summary['macro_f1'])}">{summary['macro_f1']:.2%}</div>
                </div>
                <div class="card">
                    <h3>ECE (Calibration)</h3>
                    <div class="value {get_color_class(1.0 - summary['ece'])}">{summary['ece']:.4f}</div>
                </div>
            </div>
            
            <table>
                <tr><th>Parameter</th><th>value</th></tr>
                <tr><td>TensorFlow Version</td><td>{summary['metadata']['tensorflow_version']}</td></tr>
                <tr><td>Device Profile Used</td><td>{summary['metadata']['device_used']}</td></tr>
                <tr><td>Classification Target Categories</td><td>{len(summary['classes'])} classes</td></tr>
                <tr><td>Test Dataset Size</td><td>{summary['test_set_size']} scan slices</td></tr>
                <tr><td>Total Evaluation Duration</td><td>{summary['duration_sec']:.2f} seconds</td></tr>
            </table>
        </div>
        
        <!-- SECTION 2: CLASS METRICS -->
        <div id="metrics" class="section">
            <h2>Per-Class Performance Report</h2>
            <table>
                <thead>
                    <tr>
                        <th>Class Diagnosis</th>
                        <th>Samples</th>
                        <th>Correct</th>
                        <th>Incorrect</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>ROC-AUC</th>
                    </tr>
                </thead>
                <tbody>"""
                
    for cls in summary["classes"]:
        class_stats = summary["per_class_performance"].get(cls, {})
        html += f"""
                    <tr>
                        <td><strong>{cls.upper()}</strong></td>
                        <td>{class_stats.get('samples', 0)}</td>
                        <td>{class_stats.get('correct', 0)}</td>
                        <td>{class_stats.get('incorrect', 0)}</td>
                        <td>{class_stats.get('precision', 0.0):.2%}</td>
                        <td>{class_stats.get('recall', 0.0):.2%}</td>
                        <td>{class_stats.get('f1_score', 0.0):.2%}</td>
                        <td>{class_stats.get('roc_auc', 0.0):.4f}</td>
                    </tr>"""
                    
    html += f"""
                </tbody>
            </table>
            <p><strong>Matthews Correlation Coefficient (MCC):</strong> {summary['mcc']:.4f} | <strong>Cohen's Kappa Score:</strong> {summary['cohen_kappa']:.4f}</p>
        </div>
        
        <!-- SECTION 3: CONFUSION MATRIX -->
        <div id="confusion" class="section">
            <h2>Confusion Matrix</h2>
            <div class="image-row">
                <div class="image-col">
                    <img src="../confusion_matrix.png" alt="Raw Confusion Matrix">
                    <p>Raw counts. Rows show targets, columns show predictions.</p>
                </div>
                <div class="image-col">
                    <img src="../normalized_confusion_matrix.png" alt="Normalized Confusion Matrix">
                    <p>Normalized matrix showing diagnostic true-positive and false-negative rates.</p>
                </div>
            </div>
        </div>
        
        <!-- SECTION 4: CURVES -->
        <div id="curves" class="section">
            <h2>ROC and Precision-Recall Curves</h2>
            <div class="image-row">
                <div class="image-col">
                    <img src="../roc_curve.png" alt="One-vs-Rest ROC Curve">
                    <p>Class-specific One-vs-Rest ROC and averages (Micro/Macro).</p>
                </div>
                <div class="image-col">
                    <img src="../pr_curve.png" alt="Precision-Recall Curve">
                    <p>Precision vs. Recall curves tracking trade-offs.</p>
                </div>
            </div>
        </div>
        
        <!-- SECTION 5: CALIBRATION -->
        <div id="calibration" class="section">
            <h2>Model Calibration Analysis</h2>
            <div class="image-row">
                <div class="image-col">
                    <img src="../calibration/reliability_diagram.png" alt="Reliability Calibration Diagram">
                    <p>Reliability plot. ECE: {summary['ece']:.4f} | MCE: {summary['mce']:.4f}</p>
                </div>
                <div class="image-col">
                    <img src="../calibration/confidence_distribution.png" alt="Confidence Distribution Histograms">
                    <p>Frequency of correct vs incorrect classification confidences.</p>
                </div>
            </div>
        </div>
        
        <!-- SECTION 6: GRAD-CAM GALLERY -->
        <div id="gradcam" class="section">
            <h2>Explainable AI: Representative Grad-CAM Activations</h2>
            <p>Visualizing convolutional feature map overlays to identify decision hotspots:</p>
            <div class="gradcam-grid">"""
            
    # Embed up to 8 Grad-CAM gallery items
    for item in gradcam_gallery[:8]:
        html += f"""
                <div class="gradcam-card">
                    <img src="{item['relative_path']}" alt="{item['filename']}">
                    <div class="title">{item['true_class'].upper()} ({item['mode'].upper()})</div>
                    <div class="info">Pred: {item['predicted_class']} | Conf: {item['confidence']:.2%}</div>
                </div>"""
                
    html += f"""
            </div>
        </div>
        
        <!-- SECTION 7: AUDITS -->
        <div id="misclassifications" class="section">
            <h2>Failure Diagnostics Dashboard</h2>
            <h3>Top 5 Highest-Confidence Misclassifications</h3>
            <table>
                <thead>
                    <tr>
                        <th>True Class</th>
                        <th>Predicted Class</th>
                        <th>Confidence</th>
                        <th>File Name</th>
                    </tr>
                </thead>
                <tbody>"""
                
    for row in misclassification_data["wrong"][:5]:
        html += f"""
                    <tr>
                        <td><span class="badge-danger">{row['true_class'].upper()}</span></td>
                        <td><strong>{row['predicted_class'].upper()}</strong></td>
                        <td>{row['confidence']:.2%}</td>
                        <td><code>{row['filename']}</code></td>
                    </tr>"""
                    
    html += f"""
                </tbody>
            </table>
            
            <h3>Top 5 Lowest-Confidence Correct Classifications</h3>
            <table>
                <thead>
                    <tr>
                        <th>True Class</th>
                        <th>Predicted Class</th>
                        <th>Confidence</th>
                        <th>File Name</th>
                    </tr>
                </thead>
                <tbody>"""
                
    for row in misclassification_data["borderline"][:5]:
        html += f"""
                    <tr>
                        <td><span class="badge-success">{row['true_class'].upper()}</span></td>
                        <td><strong>{row['predicted_class'].upper()}</strong></td>
                        <td>{row['confidence']:.2%}</td>
                        <td><code>{row['filename']}</code></td>
                    </tr>"""
                    
    html += f"""
                </tbody>
            </table>
        </div>
        
        <!-- SECTION 8: CONCLUSIONS -->
        <div id="conclusions" class="section">
            <h2>Conclusions & Diagnosis</h2>
            <p>This report documents the diagnostic capabilities of our EfficientNetB0-based classifier. </p>
            <ul>
                <li><strong>Generalization:</strong> Classification metrics reflect the model's accuracy on unseen samples.</li>
                <li><strong>Calibration Check:</strong> The reliability curves indicate if predictions represent physical probabilities. ECE of &lt; 0.1 indicates a well-calibrated classifier suitable for medical advice support.</li>
                <li><strong>Visual Explanations:</strong> Grad-CAM highlights ensure models trace anatomical tumors instead of scanning boundary artifacts.</li>
            </ul>
        </div>
    </div>
    
    <script>
        // Simple navigation highlighted state logic
        const links = document.querySelectorAll('.sidebar a');
        window.addEventListener('scroll', () => {{
            let current = '';
            document.querySelectorAll('.section').forEach(sec => {{
                const top = sec.offsetTop;
                if (pageYOffset >= top - 60) {{
                    current = sec.getAttribute('id');
                }}
            }});
            links.forEach(a => {{
                a.classList.remove('active');
                if (a.getAttribute('href').substring(1) === current) {{
                    a.classList.add('active');
                }}
            }});
        }});
    </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    logger.info(f"HTML evaluation report generated at: {output_path.resolve()}")

def generate_markdown_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Generates a detailed summary Markdown report.

    Args:
        summary (Dict[str, Any]): Stats summary dictionary.
        output_path (Path): Target path to write MD report.
    """
    md = f"""# Model Evaluation Summary - Brain MRI Tumor Classification

This document compiles the quantitative evaluation parameters for the Brain MRI Tumor Classifier.

---

## 1. Executive Performance Metrics

- **Test Accuracy**: {summary['accuracy']:.2%}
- **Balanced Accuracy**: {summary['balanced_accuracy']:.2%}
- **Matthews Correlation Coefficient (MCC)**: {summary['mcc']:.4f}
- **Cohen's Kappa Score**: {summary['cohen_kappa']:.4f}
- **Expected Calibration Error (ECE)**: {summary['ece']:.4f}
- **Maximum Calibration Error (MCE)**: {summary['mce']:.4f}

---

## 2. Per-Class Metrics Breakdowns

| Class Diagnosis | Test Samples | Correct | Incorrect | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for cls in summary["classes"]:
        class_stats = summary["per_class_performance"].get(cls, {})
        md += f"| {cls.upper()} | {class_stats.get('samples', 0)} | {class_stats.get('correct', 0)} | {class_stats.get('incorrect', 0)} | {class_stats.get('precision', 0.0):.2%} | {class_stats.get('recall', 0.0):.2%} | {class_stats.get('f1_score', 0.0):.2%} | {class_stats.get('roc_auc', 0.0):.4f} |\n"

    md += f"""
---

## 3. Metadata Parameters

- **TensorFlow Engine**: v{summary['metadata']['tensorflow_version']}
- **Hardware Profile**: {summary['metadata']['device_used']}
- **Random Reproducibility Seed**: {summary['metadata']['random_seed']}
- **Total Diagnostic Run Time**: {summary['duration_sec']:.2f} seconds
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
        
    logger.info(f"Markdown evaluation report generated at: {output_path.resolve()}")

def run_evaluation() -> None:
    """Executes the full evaluation, calibration, explainability, and report generating pipeline."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("           STARTING MODEL EVALUATION PIPELINE           ")
    logger.info("=" * 60)

    # Make target directories
    eval_dir = Path(config.OUTPUT_DIR) / "evaluation"
    calib_dir = eval_dir / "calibration"
    gradcam_dir = eval_dir / "gradcam"
    reports_dir = eval_dir / "reports"

    eval_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check Model exists
    best_model_path = Path(config.MODEL_PATH)
    if not best_model_path.exists():
        logger.error(
            f"Trained model '{best_model_path.resolve()}' not discovered. "
            "Skipping actual evaluation loops. Generating template placeholder report files."
        )
        # Create empty placeholder files to verify pipeline architecture
        dummy_summary = {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "ece": 0.0,
            "mce": 0.0,
            "mcc": 0.0,
            "cohen_kappa": 0.0,
            "test_set_size": 0,
            "duration_sec": 0.0,
            "classes": ["glioma", "meningioma", "notumor", "pituitary"],
            "per_class_performance": {},
            "metadata": get_system_metadata()
        }
        
        with open(reports_dir / "evaluation_summary.json", "w") as f:
            json.dump(dummy_summary, f, indent=4)
            
        generate_html_report(dummy_summary, {"wrong": [], "borderline": []}, [], reports_dir / "evaluation_report.html")
        generate_markdown_report(dummy_summary, reports_dir / "evaluation_report.md")
        return

    # Load Model — must pass custom_objects for MRIAugmentationPipeline and RandomShear
    # (CRITICAL-05: without custom_objects, load_model raises ValueError: Unknown layer)
    logger.info(f"Loading best model from {best_model_path.resolve()}...")
    try:
        model = load_model_robustly(best_model_path)
        logger.info("Successfully loaded pre-trained model.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return

    # 2. Load Dataset
    logger.info(f"Scanning dataset folders under: {config.DATASET_PATH}")
    loader = MRIDatasetLoader(config.DATASET_PATH)
    metadata = loader.scan_dataset()

    has_test = "test" in metadata and not metadata["test"].empty
    if not has_test:
        logger.warning(
            "Test split is empty or missing. "
            "Please populate 'dataset/test/' subfolders before running evaluation. "
            "Terminating gracefully."
        )
        dummy_summary = {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "ece": 0.0,
            "mce": 0.0,
            "mcc": 0.0,
            "cohen_kappa": 0.0,
            "test_set_size": 0,
            "duration_sec": 0.0,
            "classes": loader.classes if loader.classes else ["glioma", "meningioma", "notumor", "pituitary"],
            "per_class_performance": {},
            "metadata": get_system_metadata()
        }
        with open(reports_dir / "evaluation_summary.json", "w") as f:
            json.dump(dummy_summary, f, indent=4)
        generate_html_report(dummy_summary, {"wrong": [], "borderline": []}, [], reports_dir / "evaluation_report.html")
        generate_markdown_report(dummy_summary, reports_dir / "evaluation_report.md")
        return

    test_df = metadata["test"]
    logger.info(f"Found {len(test_df)} valid scans in test dataset.")

    # 3. Create tf.data inference pipeline
    file_paths = test_df["file_path"].tolist()
    class_to_idx = {cls: idx for idx, cls in enumerate(loader.classes)}
    y_true = np.array([class_to_idx[cls] for cls in test_df["class"]])

    # Load and preprocess all test images to memory for evaluation metrics computation
    # CRITICAL-06: Guard against cv2.imread returning None for unreadable files.
    # HIGH-08: cv2.resize expects (width, height), not (height, width).
    logger.info("Loading and preprocessing test images...")
    X_test_list = []
    valid_indices = []
    for i, fp in enumerate(file_paths):
        img_raw = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            logger.warning(f"Could not read image (skipping): {fp}")
            continue
        img_resized = cv2.resize(img_raw, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        img_tensor = img_resized.astype(np.float32) / 255.0
        # Convert to RGB (3 channels)
        img_rgb = np.stack([img_tensor] * 3, axis=-1)
        X_test_list.append(img_rgb)
        valid_indices.append(i)

    if not X_test_list:
        logger.error("No test images could be read. Aborting evaluation.")
        return

    # Sync y_true and test_df to exclude any skipped images
    y_true = y_true[valid_indices]
    test_df = test_df.iloc[valid_indices].reset_index(drop=True)
    file_paths = [file_paths[i] for i in valid_indices]

    X_test = np.array(X_test_list)
    logger.info(f"Loaded input tensors shape: {X_test.shape}")

    # 4. Generate predictions
    logger.info("Running inference predictions...")
    y_prob = model.predict(X_test, batch_size=config.BATCH_SIZE)
    y_pred = np.argmax(y_prob, axis=1)

    # 5. Core Metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    cohen_kappa = float(cohen_kappa_score(y_true, y_pred))

    # Detailed report from sklearn
    cls_rep = classification_report(y_true, y_pred, target_names=loader.classes, output_dict=True)
    macro_f1 = float(cls_rep["macro avg"]["f1-score"])

    # 6. Confusion Matrix Plots & CSV
    cm_df, cm_norm_df = plot_confusion_matrices(y_true, y_pred, loader.classes, eval_dir)
    
    # 7. Curves Generation (ROC & PR)
    roc_aucs, pr_aucs = plot_curves(y_true, y_prob, loader.classes, eval_dir)

    # 8. Calibration Metrics
    ece, mce, bin_accs, bin_confs, bin_sizes = calculate_calibration_metrics(y_true, y_prob)
    plot_reliability_diagram(bin_accs, bin_confs, ece, mce, calib_dir / "reliability_diagram.png")
    plot_confidence_distribution(y_true, y_prob, loader.classes, calib_dir / "confidence_distribution.png")

    # 9. Error Audits & Dashboard
    confidences = np.max(y_prob, axis=1)
    is_correct = (y_pred == y_true)
    
    # Record metadata
    test_df["predicted_class"] = [loader.classes[idx] for idx in y_pred]
    test_df["true_class"] = test_df["class"]
    test_df["confidence"] = confidences
    test_df["is_correct"] = is_correct

    # Identify wrong predictions
    wrong_cases = test_df[~is_correct].sort_values(by="confidence", ascending=False)
    # Identify lowest confidence correct predictions
    borderline_cases = test_df[is_correct].sort_values(by="confidence", ascending=True)

    # Format dashboard outputs
    dashboard_wrong = []
    for _, row in wrong_cases.head(20).iterrows():
        dashboard_wrong.append({
            "filename": row["filename"],
            "true_class": row["true_class"],
            "predicted_class": row["predicted_class"],
            "confidence": float(row["confidence"]),
            "file_path": row["file_path"]
        })
        
    dashboard_borderline = []
    for _, row in borderline_cases.head(20).iterrows():
        dashboard_borderline.append({
            "filename": row["filename"],
            "true_class": row["true_class"],
            "predicted_class": row["predicted_class"],
            "confidence": float(row["confidence"]),
            "file_path": row["file_path"]
        })

    # Save CSV Dashboard
    dashboard_df = pd.concat([wrong_cases.head(20), borderline_cases.head(20)], ignore_index=True)
    dashboard_df.to_csv(reports_dir / "misclassification_dashboard.csv", index=False)
    
    # Generate HTML failure dashboard segment
    dashboard_html_segment = """
    <html>
    <head>
        <title>Misclassification Diagnostics Dashboard</title>
        <style>
            body { font-family: sans-serif; margin: 20px; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 30px; }
            th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
            th { background-color: #f2f2f2; }
            .badge-wrong { color: red; font-weight: bold; }
            .badge-correct { color: green; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Diagnostic Audits & Borderline Classifications</h1>
        <h2>Top 20 Highest-Confidence Wrong Predictions</h2>
        <table>
            <tr><th>Filename</th><th>True Class</th><th>Predicted Class</th><th>Confidence</th></tr>"""
    for item in dashboard_wrong:
        dashboard_html_segment += f"<tr><td>{item['filename']}</td><td><span class=\"badge-wrong\">{item['true_class']}</span></td><td>{item['predicted_class']}</td><td>{item['confidence']:.2%}</td></tr>"
    dashboard_html_segment += """</table>
        <h2>Top 20 Lowest-Confidence Correct Predictions</h2>
        <table>
            <tr><th>Filename</th><th>True Class</th><th>Predicted Class</th><th>Confidence</th></tr>"""
    for item in dashboard_borderline:
        dashboard_html_segment += f"<tr><td>{item['filename']}</td><td><span class=\"badge-correct\">{item['true_class']}</span></td><td>{item['predicted_class']}</td><td>{item['confidence']:.2%}</td></tr>"
    dashboard_html_segment += "</table></body></html>"
    
    with open(reports_dir / "misclassification_dashboard.html", "w", encoding="utf-8") as f:
        f.write(dashboard_html_segment)

    # 10. Grad-CAM gallery generation
    logger.info("Generating Grad-CAM overlays for predictions...")
    gradcam_gallery = generate_gradcam_gallery(model, test_df, loader.classes, gradcam_dir)

    # 11. Per-Class summary details
    per_class_summary = {}
    for i, cls in enumerate(loader.classes):
        cls_mask = (y_true == i)
        cls_pred_mask = (y_pred == i)
        
        samples_count = int(np.sum(cls_mask))
        correct_count = int(np.sum((y_true == i) & (y_pred == i)))
        incorrect_count = samples_count - correct_count
        
        per_class_summary[cls] = {
            "samples": samples_count,
            "correct": correct_count,
            "incorrect": incorrect_count,
            "precision": float(cls_rep[cls]["precision"]),
            "recall": float(cls_rep[cls]["recall"]),
            "f1_score": float(cls_rep[cls]["f1-score"]),
            "roc_auc": float(roc_aucs.get(cls, 0.0)),
            "pr_auc": float(pr_aucs.get(cls, 0.0))
        }

    # Compile overall summary JSON
    duration = time.time() - start_time
    summary_json = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "ece": ece,
        "mce": mce,
        "mcc": mcc,
        "cohen_kappa": cohen_kappa,
        "test_set_size": len(y_true),
        "duration_sec": duration,
        "classes": loader.classes,
        "per_class_performance": per_class_summary,
        "metadata": get_system_metadata()
    }

    # Save summary JSON
    with open(reports_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary_json, f, indent=4)
    logger.info(f"Evaluation summary JSON exported to {reports_dir / 'evaluation_summary.json'}")

    # Generate complete HTML & MD reports
    generate_html_report(summary_json, {"wrong": dashboard_wrong, "borderline": dashboard_borderline}, gradcam_gallery, reports_dir / "evaluation_report.html")
    generate_markdown_report(summary_json, reports_dir / "evaluation_report.md")

    logger.info("=" * 60)
    logger.info(f"EVALUATION PIPELINE FINISHED SUCCESSFULLY IN {duration:.2f} SECONDS.")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_evaluation()
