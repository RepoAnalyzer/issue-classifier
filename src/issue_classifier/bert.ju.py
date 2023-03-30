# %%
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm.notebook import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

# %%
df = pd.read_csv("data/title_conference.csv")
df.head()

"""%%
Заметим, что классы несбалансированы:
%%"""

df.Conference.value_counts()

"""%%
## Encoding labels
%%"""

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
labels_encoded = le.fit(df.Conference).classes_
labels_dict = {l: i for (i, l) in enumerate(labels_encoded)}
df["label"] = df.Conference.replace(labels_dict)
df


"""%%
## Train and Vallidation Split
Because dataset have unbalanced classes, we split the data in stratified fashion.
%%"""

from sklearn.model_selection import train_test_split  # noqa: 402

X_train, X_val, y_train, y_val = train_test_split(
    df.index.values,
    df.label.values,
    test_size=0.15,
    random_state=42,
    stratify=df.label.values,
)

df.data_type = ["not_set"] * df.shape[0]

df.loc[X_train, "data_type"] = "train"
df.loc[X_val, "data_type"] = "val"

df.groupby(["Conference", "label", "data_type"]).count()

"""%%
## BertTokenizer
%%"""
BERT_MODEL_TYPE = "bert-base-uncased"


tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_TYPE, do_lower_case=True)


def batch_encode_plus(data):
    return tokenizer.batch_encode_plus(
        data.Title.values,
        # Sequences will be encoded with th especial tokens relative to their model.
        add_special_tokens=True,
        # Return attention mask according to specific tokenizer.
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,  # Limit just in case.
        return_tensors="pt",  # Return pytorch compatible tensors.
    )


train_data = df[df["data_type"] == "train"]
encoded_data_train = batch_encode_plus(train_data)

val_data = df[df["data_type"] == "val"]
encoded_data_val = batch_encode_plus(val_data)

input_ids_train = encoded_data_train["input_ids"]
attention_masks_train = encoded_data_train["attention_mask"]
labels_train = torch.tensor(train_data.label.values)

input_ids_val = encoded_data_val["input_ids"]
attention_masks_val = encoded_data_val["attention_mask"]
labels_val = torch.tensor(val_data.label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

"""%%
## BERT Pre-trained model
%%"""

model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_TYPE,
    num_labels=len(labels_encoded),
    output_attentions=False,
    output_hidden_states=False,
)

"""%%
## Data Loaders
Let's combine a dataset and a sampler to data loader that provides an iterable
over the given dataset.
%%"""

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler  # noqa: 402

BATCH_SIZE = 3

dataloader_train = DataLoader(
    dataset_train, sampler=RandomSampler(dataset_train), batch_size=BATCH_SIZE
)

dataloader_val = DataLoader(
    dataset_val, sampler=RandomSampler(dataset_val), batch_size=BATCH_SIZE
)

"""%%
## Optimizer & Scheduler
1. We should define parameters to optimize in iterable.
2. Then specify optmizer-specific options such as epochs, learning_rate...
3. Create a schedule with a learning rate that first inreases linearly from
0 to the initial learning rate set in the optimizer (a.k.a. warm up period) and then
decreases linearly from the initial learning rate to 0.
%%"""

from transformers import AdamW, get_linear_schedule_with_warmup  # noqa: 402

LEARNING_RATE = 1e-5
EPSILON = 1e-8
EPOCHS = 5  # Depends on dataset.

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train) * EPOCHS
)

"""%%
## Performance metrics
We will use f1 and accuracy per class.
%%"""

import numpy as np  # noqa: 402
from sklearn.metrics import f1_score  # noqa: 402


def get_f1_score(predictions, labels):
    predictions_flattened = np.argmax(predictions, axis=1).flatten()
    labels_flattened = labels.flatten()

    return f1_score(labels_flattened, predictions_flattened, average="weighted")


# %%
def accuracy_per_class(predictions, labels):
    # Inverse the dictionary.
    labels_lookup_table = {v: k for k, v in labels_dict.items()}

    predictions_flattened = np.argmax(predictions, axis=1).flatten()
    labels_flattened = labels.flatten()

    for label in np.unique(labels_flattened):
        y_predicted = predictions_flattened[labels_flattened == label]
        y_true = labels_flattened[labels_flattened == label]

        print(f"Class: {labels_lookup_table[label]}")
        print(f"Accuracy: {len(y_predicted[y_predicted==label])}/{len(y_true)}")


"""%%
### Training loop
%%"""

import random  # noqa: 402

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(device)

# %%
from pathlib import Path  # noqa: 402
from torch import nn  # noqa: 402

Path("models").mkdir(parents=True, exist_ok=True)


def map_batch_to_inputs(batch):
    return {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "labels": batch[2],
    }


def evaluate(dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = map_batch_to_inputs(batch)

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, EPOCHS + 1)):
    model.train()

    loss_train_total = 0

    progress_bar = tqdm(
        dataloader_train, desc="Epoch {:1d}".format(epoch), leave=False, disable=False
    )

    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = map_batch_to_inputs(batch)

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix(
            {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
        )

    torch.save(model.state_dict(), f"models/finetuned_BERT_epoch_{epoch}.model")

    tqdm.write(f"\nEpoch {epoch}")

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f"Training loss: {loss_train_avg}")

    val_loss, predictions, true_vals = evaluate(dataloader_val)
    val_f1 = get_f1_score(predictions, true_vals)
    tqdm.write(f"Validation loss: {val_loss}")
    tqdm.write(f"F1 Score (weighted): {val_f1}")

"""%%
## Loading and evaluating the model
%%"""

model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_TYPE,
    num_labels=len(labels_encoded),
    output_attentions=False,
    output_hidden_states=False,
)

model.to(device)

# %%
model.load_state_dict(
    torch.load("models/finetuned_BERT_epoch_1.model", map_location=torch.device("cuda"))
)

# %%
_, predictions, true_vals = evaluate(dataloader_val)
accuracy_per_class(predictions, true_vals)
