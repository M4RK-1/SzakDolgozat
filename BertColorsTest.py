import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding


def preprocess_function(examples):
    texts = [example["utterances"] for example in examples]
    return tokenizer(texts, truncation=True, padding='max_length', max_length=512)


# Load the CSV dataset
df = pd.read_csv("Data/ColorsCSV/Raw/colors.csv")

# Split the dataframe
train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the datasets
train_encodings = preprocess_function(train_df.to_dict('records'))
eval_encodings = preprocess_function(eval_df.to_dict('records'))

# Convert color labels to numeric values
colors = df['color'].unique().tolist()
train_labels = train_df['color'].apply(colors.index).tolist()
eval_labels = eval_df['color'].apply(colors.index).tolist()


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


# Evaluation
def compute_accuracy(predictions, labels):
    predicted_labels = predictions.argmax(axis=1)
    correct_predictions = (predicted_labels == labels).sum()
    accuracy = correct_predictions / len(labels)
    return accuracy


# Accuracy on training dataset
train_predictions = trainer.predict(train_dataset).predictions
train_accuracy = compute_accuracy(train_predictions, train_labels)
print(f"Accuracy on training dataset: {train_accuracy:.4f}")

# Accuracy on evaluation dataset
eval_predictions = trainer.predict(eval_dataset).predictions
eval_accuracy = compute_accuracy(eval_predictions, eval_labels)
print(f"Accuracy on eval dataset: {eval_accuracy:.4f}")
