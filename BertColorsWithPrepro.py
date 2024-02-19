import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding

finetune = False


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def compute_accuracy(predictions, labels):
    predicted_labels = predictions.argmax(axis=1)
    correct_predictions = (predicted_labels == labels).sum()
    accuracy = correct_predictions / len(labels)
    return accuracy


all_features = pd.read_csv("Data/ColorsCSV/features.csv", header=None)
all_labels = pd.read_csv("Data/ColorsCSV/all_labels.csv", header=None)
cut_labels = pd.read_csv("Data/ColorsCSV/cut_labels.csv", header=None)
unique_labels = pd.read_csv("Data/ColorsCSV/unique_labels.csv", header=None)
use_features = all_features

# get the number of available labels
num_known_labels = cut_labels.shape[0]

# get the cut features
cut_features = use_features.iloc[:num_known_labels]
use_features.drop(cut_features.index, inplace=True)

# Convert color labels to numeric values
colors = unique_labels.iloc[:, 0].unique().tolist()

cut_labels_numer = cut_labels.iloc[:, 0].apply(lambda x: colors.index(x))

# combine the cut features and labels
cut_data = pd.concat([cut_features, cut_labels_numer], axis=1)

# split the features into training and evaluation sets
train_data, eval_data = train_test_split(cut_data, test_size=0.2)

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# tokenize the datasets
train_encodings = tokenizer(train_data.iloc[:, 0].tolist(), truncation=True, padding='max_length', max_length=512)
eval_encodings = tokenizer(eval_data.iloc[:, 0].tolist(), truncation=True, padding='max_length', max_length=512)

# Convert color labels to lists
train_labels = train_data.iloc[:, 1].tolist()
eval_labels = eval_data.iloc[:, 1].tolist()

train_dataset = CustomDataset(train_encodings, train_labels)
eval_dataset = CustomDataset(eval_encodings, eval_labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(colors))

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=15,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()

train_predictions = trainer.predict(train_dataset).predictions
train_accuracy = compute_accuracy(train_predictions, train_labels)
print(f"Accuracy on training dataset: {train_accuracy:.4f}")

# Use the trained model to predict the color of the remaining features
additional_features = use_features.iloc[:100]
use_features.drop(additional_features.index, inplace=True)

# Tokenize the additional features
additional_encodings = tokenizer(additional_features.iloc[:, 0].tolist(), truncation=True, padding='max_length',
                                 max_length=512)

# Create a dataset for the additional features
additional_dataset = CustomDataset(additional_encodings, [0] * len(additional_features))

# Get predictions for the additional dataset
additional_predictions = trainer.predict(additional_dataset).predictions

# Map predicted indices to color labels
predicted_labels = [colors[prediction] for prediction in additional_predictions.argmax(axis=1)]
predicted_labels = pd.Series(predicted_labels, name="color")
predicted_labels = predicted_labels.apply(colors.index).tolist()

additional_dataset = CustomDataset(additional_encodings, predicted_labels)

trainer.save_model("fine_tuned_bert")
model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_bert")

# Retrain the model with the additional dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=additional_dataset,
    eval_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# Evaluate the model's performance
# Accuracy on training dataset
train_predictions = trainer.predict(train_dataset).predictions
train_accuracy = compute_accuracy(train_predictions, train_labels)
print(f"Accuracy on training dataset: {train_accuracy:.4f}")