import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,RobertaForMaskedLM,RobertaConfig, RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve
)
from scipy.stats import spearmanr, norm
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from collections import Counter

# =========================
# Custom Tokenizer and Loader
# =========================
class BioTokenizer(RobertaTokenizer):
    def __init__(self, ksize=1, stride=1, include_bos=False, include_eos=False, **kwargs):
        super().__init__(**kwargs)
        self.ksize = ksize
        self.stride = stride
        self.include_bos = include_bos
        self.include_eos = include_eos

    def tokenize(self, t, **kwargs):
        t = t.upper()
        if self.ksize == 1:
            toks = list(t)
        else:
            toks = [
                t[i : i + self.ksize]
                for i in range(0, len(t), self.stride)
                if len(t[i : i + self.ksize]) == self.ksize
            ]
        if len(toks) > 0 and len(toks[-1]) < self.ksize:
            toks = toks[:-1]
        if self.include_bos:
            toks = ["S"] + toks
        if self.include_eos:
            toks = toks + ["/S"]
        return toks

def load_tokenizer(model_name):
    tokenizer = BioTokenizer.from_pretrained(model_name, max_len=128)
    tokenizer.padding_side = "right"
    # Define special tokens
    cls_token = "S"
    pad_token = "P"
    sep_token = "/S"
    unk_token = "N"
    mask_token = "M"
    G_token = "G"
    A_token = "A"
    C_token = "C"
    T_token = "T"
    token_ids = tokenizer.convert_tokens_to_ids(
        [
            cls_token,
            pad_token,
            sep_token,
            unk_token,
            mask_token,
            G_token,
            A_token,
            C_token,
            T_token,
        ]
    )
    tokenizer.cls_token_id = token_ids[0]
    tokenizer.pad_token_id = token_ids[1]
    tokenizer.sep_token_id = token_ids[2]
    tokenizer.unk_token_id = token_ids[3]
    tokenizer.mask_token_id = token_ids[4]
    tokenizer.bos_token_id = token_ids[0]
    tokenizer.eos_token_id = token_ids[2]
    return tokenizer

# =========================
# Dataset Class
# =========================
class DoublingTimeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels  # continuous values for regression

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# =========================
# Metrics for Regression
# =========================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    r2 = r2_score(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)
    return {"rmse": rmse, "r2": r2, "mae": mae, "spearman_rho": spearman_corr}

# =========================
# Function to Extract CLS Embeddings
# =========================
def get_embeddings(model, dataset, batch_size=8):
    model.config.output_hidden_states = True
    model.eval()
    embeddings = []
    dataloader = DataLoader(dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # Hidden states from all transformer layers
            # Extract CLS token embedding from last hidden layer
            cls_embeds = hidden_states[-1][:, 0, :]  # [batch_size, hidden_size]
            embeddings.append(cls_embeds.cpu().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings

# =========================
# Training Pipeline (Regression)
# =========================
def train_regression_model(
    train_file, val_file, test_file, model_dir, output_dir, batch_size=8, epochs=8
):
    tokenizer = load_tokenizer(model_dir)

    # Load CSVs (must have columns: 'sequence', 'log_dob_h', 'assembly_id')
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    train_encodings = tokenizer(
        train_df["sequence"].tolist(),
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        max_length=128,
    )
    val_encodings = tokenizer(
        val_df["sequence"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    test_encodings = tokenizer(
        test_df["sequence"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    train_dataset = DoublingTimeDataset(train_encodings, train_df["log_dob_h"].values)
    val_dataset = DoublingTimeDataset(val_encodings, val_df["log_dob_h"].values)
    test_dataset = DoublingTimeDataset(test_encodings, test_df["log_dob_h"].values)

    # Load a sequence classification model (for regression, num_labels=1)
    model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=1)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=1,
        learning_rate=1e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the regression model
    trainer.train()
    # Save the best model
    trainer.save_model(os.path.join(output_dir, "best_model"))

    # Evaluate on test set
    test_results = trainer.predict(test_dataset)
    print("Regression metrics on test set:", test_results.metrics)

    # Return trainer, model, datasets, dataframes, and preds
    return (
        trainer,
        model,
        train_dataset,
        test_dataset,
        train_df,
        test_df,
        trainer.predict(train_dataset).predictions.flatten(),
        trainer.predict(test_dataset).predictions.flatten(),
    )

# =========================
# DeLong's Test Functions
# =========================
def compute_midrank(x):
    """Compute midranks for DeLong’s test."""
    sorted_x = np.sort(x)
    unique_x = np.unique(sorted_x)
    midranks = np.zeros(len(x))
    for val in unique_x:
        indices = np.where(x == val)[0]
        midrank = 0.5 * (np.min(indices) + np.max(indices)) + 1
        midranks[indices] = midrank
    return midranks

def delong_roc_variance(ground_truth, predictions):
    """
    Compute AUC and its variance via DeLong’s method.
    """
    # Sort by descending predicted score
    order = np.argsort(-predictions)
    label_ordered = ground_truth[order]
    preds_ordered = predictions[order]

    pos_preds = preds_ordered[label_ordered == 1]
    neg_preds = preds_ordered[label_ordered == 0]

    n_pos = len(pos_preds)
    n_neg = len(neg_preds)

    # Pairwise comparison matrix: 1 if pos > neg, 0.5 if pos == neg, else 0
    comparisons = np.array(
        [[1 if pos > neg else 0.5 if pos == neg else 0 for neg in neg_preds] for pos in pos_preds]
    )
    auc_value = comparisons.mean()
    V10 = comparisons.mean(axis=1)
    V01 = comparisons.mean(axis=0)
    var = (np.var(V10) / n_pos) + (np.var(V01) / n_neg)
    return auc_value, var

def delong_test(ground_truth, pred1, pred2):
    """
    Perform DeLong’s test between two sets of predicted probabilities
    (pred1, pred2) against the same ground_truth (binary 0/1).
    Returns (AUC1, AUC2, z, p_value).
    """
    auc1, var1 = delong_roc_variance(ground_truth, pred1)
    auc2, var2 = delong_roc_variance(ground_truth, pred2)
    z = (auc1 - auc2) / np.sqrt(var1 + var2)
    p = norm.sf(abs(z)) * 2
    return auc1, auc2, z, p

# =========================
# Unsupervised Classification + ROC + DeLong
# =========================

def unsupervised_classification(
    trainer,
    model,
    train_dataset,
    test_dataset,
    train_preds,
    test_preds,
    train_df,
    test_df,
    output_dir,
    batch_size=8,
):
    """
    1. Compute CLS embeddings for train/test.
    2. For thresholds (2.6, 5, 8 hrs), fit logistic regression and compute metrics.
    3. Plot ROC curves, PR-AUC-fast, and PR-AUC-slow in a 3-panel figure.
    4. Perform DeLong’s test on AUCs between each pair of thresholds.
    """
    # Step 1: Precompute embeddings
    train_embeddings = get_embeddings(model, train_dataset, batch_size=batch_size)
    test_embeddings = get_embeddings(model, test_dataset, batch_size=batch_size)

    # Step 2: Define thresholds (log-scale)
    #   2.6 hr ≈ log(2.6)=0.955, 5 hr ≈ log(5)=1.609, 8 hr ≈ log(8)=2.079
    thresholds = {
        "2.6": np.log(2.6),
        "5": np.log(5),
        "8": np.log(8),
    }

    records = []
    roc_data = {}  # To store (true_labels, predicted_probs) for ROC & DeLong

    # Step 3: Loop over each threshold
    for label, th in thresholds.items():
        # a) True labels (1 if log_dob_h > th) and pseudo-labels for train
        true_train = (train_df["log_dob_h"] > th).astype(int)
        true_test = (test_df["log_dob_h"] > th).astype(int)
        pseudo_train = (train_preds > th).astype(int)

        # b) Combine CLS embeddings + predicted doubling time as features
        X_train = np.hstack([train_embeddings, train_preds.reshape(-1, 1)])
        X_test = np.hstack([test_embeddings, test_preds.reshape(-1, 1)])

        # c) Fit logistic regression (balanced)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, pseudo_train)

        # d) Predict on test set
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)

        # e) Compute metrics
        auc_val = roc_auc_score(true_test, probs)
        pr_s = average_precision_score(true_test, probs)
        pr_f = average_precision_score(1 - true_test, 1 - probs)
        prec = precision_score(true_test, preds)
        rec = recall_score(true_test, preds)
        f1 = f1_score(true_test, preds)

        # f) Record metrics
        records.append(
            {
                "threshold_hr": float(label),
                "log_threshold": th,
                "AUC-ROC": auc_val,
                "PR-AUC-slow": pr_s,
                "PR-AUC-fast": pr_f,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
            }
        )

        # g) Store for ROC/DeLong plotting
        roc_data[label] = (true_test.values, probs)

    # Step 4: Build & save metrics DataFrame
    metrics_df = pd.DataFrame(records)
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "classification_metrics_by_threshold.csv")
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics for all thresholds to:\n  {out_csv}")

    # =========================
    # Three-panel summary plot (AUC-ROC, PR-AUC-fast, PR-AUC-slow)
    # =========================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    color_map = {"2.6": "blue", "5": "red", "8": "green"}

    # --- Panel 1: ROC curves
    for label, (y_true, probs) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color_map[label], marker="o", label=f"{label} hr: AUC={roc_auc:.2f}")
    axes[0].set_xlabel("1 - Specificity")
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_title("AUC-ROC")
    axes[0].legend(fontsize="small")
    axes[0].grid(True)

    # --- Panel 2: PR-AUC-fast curves (positive = fast)
    for label, (y_true, probs) in roc_data.items():
        precision_fast, recall_fast, _ = precision_recall_curve(1 - y_true, 1 - probs)
        pr_auc_fast = auc(recall_fast, precision_fast)
        axes[1].plot(recall_fast, precision_fast, color=color_map[label], linestyle="--", marker=".", label=f"{label} hr: AUC={pr_auc_fast:.2f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR-AUC Fast")
    axes[1].legend(fontsize="small")
    axes[1].grid(True)

    # --- Panel 3: PR-AUC-slow curves (positive = slow)
    for label, (y_true, probs) in roc_data.items():
        precision_slow, recall_slow, _ = precision_recall_curve(y_true, probs)
        pr_auc_slow = auc(recall_slow, precision_slow)
        axes[2].plot(recall_slow, precision_slow, color=color_map[label], linestyle="-", marker=".", label=f"{label} hr: AUC={pr_auc_slow:.2f}")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title("PR-AUC Slow")
    axes[2].legend(fontsize="small")
    axes[2].grid(True)

    plt.tight_layout()
    three_panel_path = os.path.join(output_dir, "ROC_PRAUC_fast_slow_panel.png")
    plt.savefig(three_panel_path, dpi=300)
    plt.show()
    print(f"Saved three-panel ROC/PR-AUC plot to:\n  {three_panel_path}")

    # Step 5: DeLong's test comparisons
    pairs = [("2.6", "5"), ("2.6", "8"), ("5", "8")]
    delong_results = []
    for h1, h2 in pairs:
        y_true = roc_data[h1][0]
        probs1 = roc_data[h1][1]
        probs2 = roc_data[h2][1]
        auc1, auc2, z, p = delong_test(y_true, probs1, probs2)
        delong_results.append(
            {"Comparison": f"{h1} vs {h2}", "AUC1": auc1, "AUC2": auc2, "z-score": z, "p-value": p}
        )

    delong_df = pd.DataFrame(delong_results)
    print(delong_df)
    #import ace_tools as tools
    #tools.display_dataframe_to_user(name="DeLong Test Results", dataframe=delong_df)

    return metrics_df, delong_df


# =========================
# Example Usage
# =========================
# Replace these paths with your actual file paths
train_file = "data_preprocess/temp_corr/iso_rib_temp_mod_train.csv"
val_file = "data_preprocess/temp_corr/iso_rib_temp_mod_val.csv"
test_file = "data_preprocess/temp_corr/iso_rib_temp_mod_test.csv"
model_dir = "LookingGlass-2"         # Where your base model is saved
output_dir = "finetune_model_classify"  # Where you want to save outputs

# 1) Train the regression model to predict doubling time
trainer, model, train_dataset, test_dataset, train_df, test_df, train_preds, test_preds = train_regression_model(
    train_file, val_file, test_file, model_dir, output_dir, batch_size=8, epochs=8
)

# 2) Perform unsupervised classification with ROC + DeLong integration
metrics_df, delong_df = unsupervised_classification(
    trainer,
    model,
    train_dataset,
    test_dataset,
    train_preds,
    test_preds,
    train_df,
    test_df,
    output_dir,
    batch_size=8,
)

