import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding

# Define the batch size for iterative training
step = 100

# Evaluate by the model or the actual data
train_model_cheat = False


# Read the data
def read_data(file_names):
    dataframes = []
    for file_name in file_names:
        df = pd.read_csv(file_name, delimiter=';', header=None)
        df.columns = range(df.shape[1])
        dataframes.append(df)
    return dataframes


# Preprocess the data
def preprocess_data(dataframes):
    df = pd.concat(dataframes[:2], axis=1)
    nan_mask = df.isna().any(axis=1)
    df_known = df[~nan_mask]
    df_predict = df[nan_mask]
    feature = df_known.iloc[:, :-1]
    label = df_known.iloc[:, -1]
    return feature, label, df_predict


# Train the model
def train_model(x_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    return model


# Evaluate the model's performance
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


def main():
    # Python things (remove later)
    cheat_counter = 0

    # Read the full dataset
    full_data = pd.read_csv("Data/ColorsCSV/features.csv")

    # Read the correct labels
    correct_labels = pd.read_csv("Data/ColorsCSV/all_labels.csv")

    # Read the data
    file_names = ["Data/ColorsCSV/features.csv", "Data/ColorsCSV/cut_labels.csv", "Data/ColorsCSV/unique_labels.csv"]
    dataframes = read_data(file_names)

    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Accuracy on training dataset
    train_predictions = trainer.predict(train_dataset).predictions
    train_accuracy = compute_accuracy(train_predictions, train_labels)
    print(f"Accuracy on training dataset: {train_accuracy:.4f}")

    # Accuracy on evaluation dataset
    eval_predictions = trainer.predict(eval_dataset).predictions
    eval_accuracy = compute_accuracy(eval_predictions, eval_labels)
    print(f"Accuracy on eval dataset: {eval_accuracy:.4f}")

'''
    # Start iterative training
    while len(df_predict) > 0:
        # Extract the next batch of unknown instances
        correct_stepsize = min(step, len(df_predict))
        new_data_step = df_predict.iloc[:correct_stepsize, :-1]
        df_predict = df_predict.iloc[correct_stepsize:, :]

        # Make predictions on the next batch of unknown instances
        predictions = model.predict(new_data_step)

        # Concatenate features with known data
        new_feature_known = pd.concat([feature_known, new_data_step])

        # Concatenate predictions with known data
        if train_model_cheat:
            # update the iteration counter
            cheat_counter += 1

            # calculate the start position of the next batch
            start_pos = step * cheat_counter - 1

            # concatenate the correct labels with the known data
            new_label_known = pd.concat([
                label_known,
                pd.Series(correct_labels.iloc[start_pos: start_pos + correct_stepsize, 0])
            ])
        else:
            # concatenate the predictions with the known data
            new_label_known = pd.concat([label_known, pd.Series(predictions)])

        # Split the updated known data into training and testing sets
        new_feature_train, new_feature_test, new_label_train, new_label_test = train_test_split(new_feature_known,
                                                                                                new_label_known,
                                                                                                test_size=0.2)

        # Retrain the model with updated data
        model = train_model(new_feature_train, new_label_train)

        # Evaluate the model's performance (optional)
        evaluate_model(model, new_feature_test, new_label_test)

        # Update known data and unknown data for the next iteration
        feature_known, label_known = new_feature_known, new_label_known

    # Make predictions with the final model
    all_predictions = model.predict(full_data)

    # Evaluate predictions based on correct labels
    evaluate_predictions(all_predictions, correct_labels)
'''

def evaluate_predictions(predictions, correct_labels):
    # Ensure correct length and alignment
    if len(predictions) != len(correct_labels):
        raise ValueError("Length mismatch between predictions and correct labels")

    # Calculate accuracy
    accuracy = accuracy_score(correct_labels, predictions)
    print("Evaluation Results:")
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
