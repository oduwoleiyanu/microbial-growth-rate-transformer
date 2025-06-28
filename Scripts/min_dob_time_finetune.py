import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification,RobertaForMaskedLM,Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import numpy as np

# Define custom tokenizer (same as before)
class BioTokenizer(RobertaTokenizer):
    def __init__(self, ksize=1, stride=1, include_bos=False, include_eos=False, **kwargs):
        super().__init__(**kwargs)
        self.ksize = ksize
        self.stride = stride
        self.include_bos = include_bos
        self.include_eos = include_eos

    def tokenize(self, t, **kwargs):
        include_bos = self.include_bos if self.include_bos is not None else include_bos
        include_eos = self.include_eos if self.include_eos is not None else include_eos
        t = t.upper()
        if self.ksize == 1:
            toks = list(t)
        else:
            toks = [t[i:i + self.ksize] for i in range(0, len(t), self.stride) if len(t[i:i + self.ksize]) == self.ksize]
        if len(toks) > 0 and len(toks[-1]) < self.ksize:
            toks = toks[:-1]
        if include_bos:
            toks = ['S'] + toks
        if include_eos:
            toks = toks + ['/S']
        return toks
def load_tokenizer(model_name):
    # Load the tokenizer from the local directory
    tokenizer = BioTokenizer.from_pretrained(model_name, max_len=128)

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

    # Assign token IDs for special tokens
    token_ids = tokenizer.convert_tokens_to_ids([
        cls_token, pad_token, sep_token, unk_token, mask_token,
        G_token, A_token, C_token, T_token
    ])

    tokenizer.cls_token_id = token_ids[0]
    tokenizer.pad_token_id = token_ids[1]
    tokenizer.sep_token_id = token_ids[2]
    tokenizer.unk_token_id = token_ids[3]
    tokenizer.mask_token_id = token_ids[4]
    tokenizer.bos_token_id = token_ids[0]
    tokenizer.eos_token_id = token_ids[2]

    return tokenizer
# Load the tokenizer and model from the "LookingGlass-2" directory
tokenizer = load_tokenizer("LookingGlass-2")
model = RobertaForMaskedLM.from_pretrained("LookingGlass-2")


# Dataset class
class DoublingTimeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float) # Important: float for regression
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten() # Flatten predictions if needed
    labels = labels.flatten() #flatten label as well
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    r2 = r2_score(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions) # spearman's rank 
    pearson_corr, _ = pearsonr(labels, predictions)
    return {"rmse": rmse, "r2": r2, "mae": mae, "spearman_rho": spearman_corr,"pearson_rho": pearson_corr}

def train_model(train_file, val_file, test_file, model_dir, output_dir, batch_size=8, epochs=8):

    tokenizer = load_tokenizer(model_dir)

    # Load data
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    # Tokenize sequences
    train_encodings = tokenizer(train_df['sequence'].tolist(), truncation=True, padding=True, max_length = 128, return_overflowing_tokens = True, return_length = True)
    val_encodings = tokenizer(val_df['sequence'].tolist(), truncation=True, padding=True, max_length = 128, return_overflowing_tokens = True, return_length = True)
    test_encodings = tokenizer(test_df['sequence'].tolist(), truncation=True, padding=True, max_length = 128, return_overflowing_tokens = True, return_length = True)

    train_dataset = DoublingTimeDataset(train_encodings, train_df['log_dob_h'].values)
    val_dataset = DoublingTimeDataset(val_encodings, val_df['log_dob_h'].values)
    test_dataset = DoublingTimeDataset(test_encodings, test_df['log_dob_h'].values)

    # Load model for sequence classification (regression)
    model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=1)  # num_labels=1 for regression

    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=1,
        learning_rate=3e-5
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # the training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluate on the test set
    test_results = trainer.predict(test_dataset)
    print(test_results.metrics)

    # Save the model
    trainer.save_model(os.path.join(output_dir, "best_model"))

    # Predict and save test results to a CSV
    predictions = test_results.predictions.flatten()  # Flatten predictions if needed
    results_df = pd.DataFrame({
        'assembly_id': test_df['assembly_id'],
        'log_dob_h': test_df['log_dob_h'],
        'predicted_dob_h': predictions
    })

    results_csv = os.path.join(output_dir, "test_predictions.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Test predictions saved to {results_csv}")


# Ribosomal sequences file
train_file = "data_preprocess/temp_corr/iso_rib_temp_mod_train.csv"
val_file = "data_preprocess/temp_corr/iso_rib_temp_mod_val.csv"
test_file = "data_preprocess/temp_corr/iso_rib_temp_mod_test.csv"
#cog cat example;  cell signalling files
#train_file = "data_preprocess/cog_cat/iso_cellsig_train.csv"
#val_file = "data_preprocess/cog_cat/iso_cellsig_val.csv"
#test_file = "data_preprocess/cog_cat/iso_cellsig_test.csv"
model_dir = "LookingGlass-2"  # Directory where your base model is saved
output_dir = "finetune_model_allnt"  # Directory to save fine-tuned model and checkpoints

if __name__ == "__main__":
    train_model(train_file, val_file, test_file, model_dir, output_dir)

