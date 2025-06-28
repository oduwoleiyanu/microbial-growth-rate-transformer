import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer,RobertaModel, RobertaConfig, RobertaForSequenceClassification, RobertaForMaskedLM, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
import numpy as np
import torch.nn as nn

# ========= Custom Tokenizer =========
class BioTokenizer(RobertaTokenizer):
    def __init__(self, ksize=1, stride=1, include_bos=False, include_eos=False, **kwargs):
        super().__init__(**kwargs)
        self.ksize = ksize
        self.stride = stride
        self.include_bos = include_bos
        self.include_eos = include_eos

    def tokenize(self, t, **kwargs):
        t = t.upper()
        toks = [t[i:i + self.ksize] for i in range(0, len(t), self.stride) if len(t[i:i + self.ksize]) == self.ksize]
        if self.include_bos:
            toks = ['S'] + toks
        if self.include_eos:
            toks = toks + ['/S']
        return toks

def load_tokenizer(model_name):
    tokenizer = BioTokenizer.from_pretrained(model_name, max_len=128)
    cls_token = "S"; pad_token = "P"; sep_token = "/S"; unk_token = "N"; mask_token = "M"
    token_ids = tokenizer.convert_tokens_to_ids([cls_token, pad_token, sep_token, unk_token, mask_token])
    tokenizer.cls_token_id = token_ids[0]
    tokenizer.pad_token_id = token_ids[1]
    tokenizer.sep_token_id = token_ids[2]
    tokenizer.unk_token_id = token_ids[3]
    tokenizer.mask_token_id = token_ids[4]
    tokenizer.bos_token_id = token_ids[0]
    tokenizer.eos_token_id = token_ids[2]
    return tokenizer

# ========= Dataset =========
class DoublingTimeDataset(Dataset):
    def __init__(self, encodings, growth_temps, labels):
        self.encodings = encodings
        self.growth_temps = growth_temps
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'growth_tmp': torch.tensor(self.growth_temps[idx], dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
        return item

    def __len__(self):
        return len(self.labels)

# ========= Custom Model =========
class RobertaWithGrowthTmp(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids=None, attention_mask=None, growth_tmp=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        combined = torch.cat([cls_output, growth_tmp.unsqueeze(1)], dim=1)
        preds = self.regressor(combined).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(preds, labels)

        return {'loss': loss, 'logits': preds}

# ========= Custom Data Collator =========
class CustomDataCollator:
    def __call__(self, features):
        return {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'growth_tmp': torch.stack([f['growth_tmp'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features])
        }

# ========= Metrics =========
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    r2 = r2_score(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)
    return {"rmse": rmse, "r2": r2, "mae": mae, "spearman_rho": spearman_corr}

# ========= Training Function =========
def train_model(train_file, val_file, test_file, model_dir, output_dir, batch_size=8, epochs=8):
    tokenizer = load_tokenizer(model_dir)
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    # Tokenization
    #train_enc = tokenizer(train_df['sequence'].tolist(), truncation=True, padding="max_length", max_length=128)
    #val_enc = tokenizer(val_df['sequence'].tolist(), truncation=True, padding="max_length", max_length=128)
    #test_enc = tokenizer(test_df['sequence'].tolist(), truncation=True, padding="max_length", max_length=128)
    # Tokenization with all 128nt
    train_enc = tokenizer(train_df['sequence'].tolist(), truncation=True, padding=True, max_length = 128, return_overflowing_tokens = True, return_length = True)
    val_enc = tokenizer(val_df['sequence'].tolist(), truncation=True, padding=True, max_length = 128, return_overflowing_tokens = True, return_length = True)
    test_enc = tokenizer(test_df['sequence'].tolist(), truncation=True, padding=True, max_length = 128, return_overflowing_tokens = True, return_length = True)

    # Datasets
    train_dataset = DoublingTimeDataset(train_enc, train_df['growth_tmp'].values, train_df['log_dob_h'].values)
    val_dataset = DoublingTimeDataset(val_enc, val_df['growth_tmp'].values, val_df['log_dob_h'].values)
    test_dataset = DoublingTimeDataset(test_enc, test_df['growth_tmp'].values, test_df['log_dob_h'].values)

    # Model
    model = RobertaWithGrowthTmp(model_dir)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=1,
        learning_rate=2e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=CustomDataCollator()
    )

    trainer.train()

    # Evaluate and save predictions
    test_results = trainer.predict(test_dataset)
    print(test_results.metrics)

    predictions = test_results.predictions.flatten()
    results_df = pd.DataFrame({
        'assembly_id': test_df['assembly_id'],
        'log_dob_h': test_df['log_dob_h'],
        'growth_tmp': test_df['growth_tmp'],
        'predicted_dob_h': predictions
    })

    results_csv = os.path.join(output_dir, "test_predictions_with_temp.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")

# ===== Example Usage =====
train_file = "data_preprocess/temp_corr/iso_rib_temp_mod_train.csv"
val_file = "data_preprocess/temp_corr/iso_rib_temp_mod_val.csv"
test_file = "data_preprocess/temp_corr/iso_rib_temp_mod_test.csv"
model_dir = "LookingGlass-2"
output_dir = "finetune_model_temp_upd"

train_model(train_file, val_file, test_file, model_dir, output_dir)

