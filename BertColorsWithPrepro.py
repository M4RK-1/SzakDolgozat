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


def evaluate_on_subset(trainer, tokenizer, colors, features, labels, num_items):
    """
    Evaluate the model on the last 'num_items' features and labels.

    Parameters:
    - trainer: The trained Trainer object.
    - tokenizer: The tokenizer used for the model.
    - colors: A list of unique color names used for converting labels to numeric values.
    - features: A DataFrame containing the features.
    - labels: A DataFrame containing the labels.
    - num_items: Number of items from the end of the dataset to evaluate on.

    Returns:
    - Accuracy of the model on the specified subset of the dataset.
    """
    # Extract the last 'num_items' features and labels
    subset_features = features.iloc[-num_items:]
    subset_labels = labels.iloc[-num_items:]

    # Tokenize the subset features
    subset_encodings = tokenizer(subset_features[0].tolist(), truncation=True, padding='max_length', max_length=512)

    # Convert the subset labels to numeric values
    subset_labels_numer = subset_labels[0].apply(lambda x: colors.index(x))

    # Create a dataset with the subset tokenized features and numeric labels
    subset_dataset = CustomDataset(subset_encodings, subset_labels_numer.tolist())

    # Predict using the trained model
    predictions = trainer.predict(subset_dataset).predictions

    # Compute accuracy
    accuracy = compute_accuracy(predictions, subset_labels_numer.tolist())

    print(f"Accuracy on the last 100 features: {accuracy:.4f}")


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

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# get the number of available labels
num_known_labels = cut_labels.shape[0]

# get the cut features
use_features, cut_features = use_features[num_known_labels:], use_features[:num_known_labels]

# Convert color labels to numeric values
uniqe_labels_num = unique_labels.iloc[:, 0].unique().tolist()

# convert the cut labels to numeric values
cut_labels_numer = cut_labels.iloc[:, 0].apply(lambda x: uniqe_labels_num.index(x))

# combine the cut features and labels
cut_data = pd.concat([cut_features, cut_labels_numer], axis=1)

# split the features into training and evaluation sets
train_data, eval_data = train_test_split(cut_data, test_size=0.2)

# tokenize the datasets
train_encodings = tokenizer(train_data.iloc[:, 0].tolist(), truncation=True, padding='max_length', max_length=512)
eval_encodings = tokenizer(eval_data.iloc[:, 0].tolist(), truncation=True, padding='max_length', max_length=512)

# Convert color labels to lists
train_labels = train_data.iloc[:, 1].tolist()
eval_labels = eval_data.iloc[:, 1].tolist()

train_dataset = CustomDataset(train_encodings, train_labels)
eval_dataset = CustomDataset(eval_encodings, eval_labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(uniqe_labels_num))

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

trainer.save_model("fine_tuned_bert")
model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_bert")

evaluate_on_subset(trainer, tokenizer, uniqe_labels_num, all_features, all_labels, 100)

batch_size = 100

while len(use_features) > 0:
    # Determine the size of the next batch
    current_batch_size = min(batch_size, len(use_features))

    # Use the trained model to predict the color of the remaining features
    use_features, additional_features = use_features[current_batch_size:], use_features[:current_batch_size]

    # Tokenize the additional features
    additional_encodings = tokenizer(additional_features.iloc[:, 0].tolist(), truncation=True, padding='max_length',
                                     max_length=512)

    # Create a dataset for the additional features
    additional_dataset = CustomDataset(additional_encodings, [0] * len(additional_features))

    # Get predictions for the additional dataset
    additional_predictions = trainer.predict(additional_dataset).predictions

    # Map predicted indices to color labels
    predicted_labels = [uniqe_labels_num[prediction] for prediction in additional_predictions.argmax(axis=1)]
    predicted_labels = pd.Series(predicted_labels, name="color")
    predicted_labels = predicted_labels.apply(uniqe_labels_num.index).tolist()

    print(str(len(additional_encodings['input_ids'])) + " " + str(len(
        predicted_labels)))

    additional_dataset = CustomDataset(additional_encodings, predicted_labels)

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

    evaluate_on_subset(trainer, tokenizer, uniqe_labels_num, all_features, all_labels, 100)

    trainer.save_model("fine_tuned_bert")
    model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_bert")

"""
train_predictions = trainer.predict(train_dataset).predictions
train_accuracy = compute_accuracy(train_predictions, train_labels)
print(f"Accuracy on training dataset: {train_accuracy:.4f}")
"""
