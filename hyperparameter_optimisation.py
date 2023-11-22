import pandas as pd
import os
import time
from typing import List

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
import torch

from sklearn.model_selection import train_test_split
import torch

from transformers import TrainingArguments, Trainer

import evaluate

from sklearn.utils.class_weight import compute_class_weight

from torch import nn
from transformers import Trainer
import optuna

from sklearn.model_selection import StratifiedKFold

import json




#Current version
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, labels, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

        # Tokenize the SMILES strings and store them
        self.encodings = self.tokenize_smiles(dataframe['SMILES'].tolist())

        # Store the labels
        self.labels = labels.tolist()

    def tokenize_smiles(self, smiles_list):
        return self.tokenizer(
            smiles_list,
            truncation=True,
            padding=True,
            max_length=None,
            return_tensors='pt'
        )

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
        return item

    def __len__(self):
        return len(self.labels)
    
class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss with class_weights=balanced from above
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=model.device, dtype=torch.float))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# Senolytic dataset - labelled
senolytics_df = pd.read_csv('list_of_compounds_for_training.csv')

training_df = senolytics_df[['SMILES', 'senolytic']]

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
config = AutoConfig.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
config.num_hidden_layers += 1
model = AutoModelForSequenceClassification.from_pretrained("DeepChem/ChemBERTa-77M-MTR", num_labels=2, problem_type = "single_label_classification")


X = training_df[['SMILES']]
y = training_df['senolytic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 


train_dataset = Dataset(X_train, y_train, tokenizer)
test_dataset = Dataset(X_test, y_test, tokenizer)




# many more parameters to experiment with https://huggingface.co/docs/transformers/v4.33.2/en/main_classes/trainer#transformers.TrainingArguments
training_args = TrainingArguments(output_dir="test_1", load_best_model_at_end=True, evaluation_strategy='epoch',
    logging_strategy="epoch", save_strategy="epoch",per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,optim="adamw_torch", num_train_epochs=10) # switch optimizer to avoid warning)




metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    metrics = ["accuracy", "recall", "precision", "f1"] #List of metrics to return
    metric={}
    for met in metrics:
       metric[met] = evaluate.load(met)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_res={}
    for met in metrics:
       metric_res[met]=metric[met].compute(predictions=predictions, references=labels)[met]
    return metric_res


# Class weights
class_weights = compute_class_weight(class_weight="balanced",classes=np.unique(y_train),y=y_train)


X = X.to_numpy()
y = y.to_numpy()



def objective(trial: optuna.Trial):
    model = AutoModelForSequenceClassification.from_pretrained("DeepChem/ChemBERTa-77M-MTR", num_labels=2, problem_type = "single_label_classification")
    training_args = TrainingArguments(
        output_dir="optuna-test",
        learning_rate=trial.suggest_loguniform("learning_rate", low=4e-5, high=0.01),
        weight_decay=trial.suggest_loguniform("weight_decay", 4e-5, 0.01),
        num_train_epochs=trial.suggest_int("num_train_epochs", low=4, high=10),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        disable_tqdm=True,
    )

    cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=62)

    # lists for this cv
    y_tests = []
    y_preds = []
    f1s = []

    for i, (train, test) in enumerate(cv.split(X, y)):
        # fit model to cv's X[train]

        train_df = pd.DataFrame(X[train], columns=['SMILES'])
        test_df =  pd.DataFrame(X[test], columns=['SMILES'])

        train_dataset = Dataset(train_df, y[train], tokenizer)
        test_dataset = Dataset(test_df, y[test], tokenizer)


        trainer = CustomTrainer(
          model=model,
          args=training_args,
          train_dataset=train_dataset,
          eval_dataset=test_dataset,
          compute_metrics=compute_metrics,
        )

        # predict on cValidation set
        result = trainer.train()
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=-1)

        # save y_test, y_pred (and y_prob) to compute confmats (& curves)
        y_tests.append(y[test])
        y_preds.append(y_pred)
        results = trainer.evaluate()
        f1s.append(results['eval_f1'])


    #final_score = metric.compute(predictions=y_pred, references=y_test)
    return sum(f1s)/len(f1s)


# We want to minimise the f1
study = optuna.create_study(study_name="hyper-parameter-search", direction="maximize")
study.optimize(func=objective, n_trials=100)
print(study.best_value)
print(study.best_params)
print(study.best_trial)

best_trial_results = {"best_value": study.best_value, "best_params": study.best_params, "best_trial": study.best_trial}

with open("best_trial.txt", "w") as fp:
    json.dump(best_trial_results, fp)  # encode dict into JSON
print("Done writing dict into .txt file")

