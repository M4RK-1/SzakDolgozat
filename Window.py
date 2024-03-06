import sys
import threading
import time

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QPushButton, QStackedWidget, QLabel, QFileDialog, QTextEdit, QComboBox, \
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QProgressBar
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, TrainerCallback


class ProgressCallback(TrainerCallback):
    def __init__(self, progress_bar, num_update_steps, update_training_status_signal):
        super().__init__()
        self.progress_bar = progress_bar
        self.num_update_steps = num_update_steps
        self.current_step = 0
        self.update_training_status_signal = update_training_status_signal

    def on_train_begin(self, args, state, control, **kwargs):
        self.update_training_status_signal.emit("Training in progress...")
    def on_step_end(self, args, state, control, **kwargs):
        self.current_step += 1
        progress = int((self.current_step / self.num_update_steps) * 100)
        progress = min(progress, 100)
        self.progress_bar.setValue(progress)

    def on_train_end(self, args, state, control, **kwargs):
        self.update_training_status_signal.emit("HIDE_LABEL")
        self.progress_bar.setValue(100)


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


class MyApp(QWidget):
    prediction_ready = pyqtSignal(list, pd.DataFrame, QWidget)
    update_training_status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.import_button1 = None
        self.imported_files_label = None
        self.combo_boxes = []
        self.stacked_widget = None
        self.dataframes = [None, None, None]
        self.df_possible_labels = None
        self.progress_bar = None
        self.training_status_label = QLabel()
        self.initUI()
        self.update_training_status.connect(self.updateTrainingStatusLabel)

    def updateTrainingStatusLabel(self, text):
        if text == "HIDE_LABEL":
            self.training_status_label.hide()
        else:
            self.training_status_label.setText(text)
            self.training_status_label.show()

    def initUI(self):
        self.setWindowTitle("SmartLabeler")
        self.setGeometry(200, 200, 1000, 700)

        layout = QVBoxLayout()
        self.stacked_widget = QStackedWidget()
        page1 = QWidget()
        page2 = QWidget()
        page3 = QWidget()

        button1 = QPushButton("Load files")
        button2 = QPushButton("Check data")
        button3 = QPushButton("Predict")

        button1.clicked.connect(self.LoadFilesPage)
        button2.clicked.connect(self.SeeLoadedPage)
        button3.clicked.connect(self.CalculationsPage)

        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)
        layout.addWidget(self.stacked_widget)

        self.import_button1 = QPushButton("Load Data File")
        self.import_button2 = QPushButton("Load Known Labels File")
        self.import_button3 = QPushButton("Load Unique Labels File")

        self.import_button1.clicked.connect(self.importData)
        self.import_button2.clicked.connect(self.importLabels)
        self.import_button3.clicked.connect(self.importLabelTypes)

        self.imported_files_label = QLabel()

        page1.layout = QVBoxLayout()
        page1.layout.addWidget(self.import_button1)
        page1.layout.addWidget(self.import_button2)
        page1.layout.addWidget(self.import_button3)
        page1.layout.addWidget(self.imported_files_label)

        page1.setLayout(page1.layout)
        self.stacked_widget.addWidget(page1)
        self.stacked_widget.addWidget(page2)
        self.stacked_widget.addWidget(page3)

        self.text_edit = QTextEdit()
        page2.layout = QVBoxLayout()
        page2.layout.addWidget(self.text_edit)
        page2.setLayout(page2.layout)

        self.progress_bar = QProgressBar(self)

        layout.addWidget(self.training_status_label)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def LoadFilesPage(self):
        self.stacked_widget.setCurrentIndex(0)

    def SeeLoadedPage(self):
        self.stacked_widget.setCurrentIndex(1)
        concatenated_df = pd.concat(self.dataframes[:2], axis=1)
        text_to_display = concatenated_df.iloc[1:].to_string(index=False)
        self.text_edit.setPlainText(text_to_display)

    def CalculationsPage(self):
        self.stacked_widget.setCurrentIndex(2)
        page3 = self.stacked_widget.widget(2)

        def train_model():
            all_features = self.dataframes[0]
            cut_labels = self.dataframes[1]
            unique_labels = self.dataframes[2]
            use_features = all_features

            # load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

            # get the number of available labels
            num_known_labels = cut_labels.shape[0]

            # get the cut features
            use_features, cut_features = use_features[num_known_labels:], use_features[:num_known_labels]

            # Convert labels to numeric values
            unique_labels_num = unique_labels.iloc[:, 0].unique().tolist()

            # convert the cut labels to numeric values
            cut_labels_numer = cut_labels.iloc[:, 0].apply(lambda x: unique_labels_num.index(x))

            # combine the cut features and labels
            cut_data = pd.concat([cut_features, cut_labels_numer], axis=1)

            # split the features into training and evaluation sets
            train_data, eval_data = train_test_split(cut_data, test_size=0.2)

            # tokenize the datasets
            train_encodings = tokenizer(train_data.iloc[:, 0].tolist(), truncation=True, padding='max_length',
                                        max_length=512)
            eval_encodings = tokenizer(eval_data.iloc[:, 0].tolist(), truncation=True, padding='max_length',
                                       max_length=512)

            # Convert color labels to lists
            train_labels = train_data.iloc[:, 1].tolist()
            eval_labels = eval_data.iloc[:, 1].tolist()

            train_dataset = CustomDataset(train_encodings, train_labels)
            eval_dataset = CustomDataset(eval_encodings, eval_labels)

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                       num_labels=len(unique_labels_num))

            training_args = TrainingArguments(
                output_dir="./results",
                learning_rate=2e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                num_train_epochs=15,
                weight_decay=0.01,
            )

            # Assume num_epochs, train_dataset, and batch_size are defined
            num_epochs = training_args.num_train_epochs
            batch_size = training_args.per_device_train_batch_size
            num_update_steps = (len(train_dataset) // batch_size + 1) * num_epochs

            progress_callback = ProgressCallback(self.progress_bar, num_update_steps, self.update_training_status)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=[progress_callback],
            )

            trainer.train()


            trainer.save_model("fine_tuned_bert")
            model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_bert")

            # Use the trained model to predict the color of the remaining features
            use_features, additional_features = use_features[100:], use_features[:100]

            # Tokenize the additional features
            additional_encodings = tokenizer(additional_features.iloc[:, 0].tolist(), truncation=True,
                                             padding='max_length',
                                             max_length=512)

            # Create a dataset for the additional features
            additional_dataset = CustomDataset(additional_encodings, [0] * len(additional_features))

            # Make predictions on the additional features
            additional_predictions = trainer.predict(additional_dataset).predictions
            predicted_labels = [unique_labels_num[prediction] for prediction in
                                additional_predictions.argmax(axis=1)]

            # Emit the signal after training is finished
            self.prediction_ready.emit(predicted_labels, use_features, page3)

        # Connect the signal to a slot that updates the UI
        self.prediction_ready.connect(self.display_predicted_labels)

        # Start training thread
        train_thread = threading.Thread(target=train_model)
        train_thread.start()

    def display_predicted_labels(self, predicted_labels, use_features, page3):
        scroll_area = QScrollArea(self)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        for i, prediction in enumerate(predicted_labels):
            h_layout = QHBoxLayout()

            value_label = QLabel(f"Value {i}: {use_features.iloc[i, 0]}")
            combo_box = QComboBox()
            combo_box.addItems(self.df_possible_labels.iloc[:, 0])
            combo_box.setCurrentText(str(prediction))
            combo_box.setMaximumWidth(150)

            self.combo_boxes.append(combo_box)

            h_layout.addWidget(value_label)
            h_layout.addWidget(combo_box)

            scroll_layout.addLayout(h_layout)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)

        correct_button = QPushButton("Correct", self)
        correct_button.clicked.connect(self.correctAllPredictions)

        if page3.layout() is None:
            layout = QVBoxLayout()
            page3.setLayout(layout)
        else:
            layout = page3.layout()

        layout.addWidget(scroll_area)
        layout.addWidget(correct_button)

    def correctAllPredictions(self):
        corrected_labels = []

        for combo_box in self.combo_boxes:
            corrected_label = combo_box.currentText()
            corrected_labels.append(corrected_label)

        # Step 3: Convert the list to a DataFrame
        corrected_labels_df = pd.DataFrame(corrected_labels)

        # Optional: Display the DataFrame, save it to a file, or perform further processing
        print(corrected_labels_df)  # For demonstration; prints the DataFrame to console

        # If you want to save this DataFrame to a CSV file:
        #corrected_labels_df.to_csv("corrected_labels.csv", index=False)

    def importData(self):
        # uncomment this line to use the file dialog
        # file_path, _ = QFileDialog.getOpenFileName(self, "Select File 1 (Data)", "",
        #                                           "CSV Files (*.csv);;All Files (*)")

        file_path = "Data/ColorsCSV/features.csv"
        if file_path:
            try:
                df = pd.read_csv(file_path, delimiter=';', header=None)
                df.columns = range(df.shape[1])
                self.dataframes[0] = df
                self.import_button1.setText(f"Data File: {file_path.split('/')[-1]} (Loaded) ✔")
            except Exception as e:
                self.imported_files_label.setText(f"Data File: Error loading ({str(e)})")

    def importLabels(self):
        # uncomment this line to use the file dialog
        # file_path, _ = QFileDialog.getOpenFileName(self, "Select File 2 (Class Labels)", "",
        #                                           "CSV Files (*.csv);;All Files (*)")

        file_path = "Data/ColorsCSV/cut_labels.csv"
        if file_path:
            try:
                df = pd.read_csv(file_path, delimiter=';', header=None)
                df.columns = range(df.shape[1])
                self.dataframes[1] = df
                self.import_button2.setText(f"Class Labels File: {file_path.split('/')[-1]} (Loaded) ✔")
            except Exception as e:
                self.imported_files_label.setText(f"Class Labels File: Error loading ({str(e)})")

    def importLabelTypes(self):
        # uncomment this line to use the file dialog
        # file_path, _ = QFileDialog.getOpenFileName(self, "Select File 3 (Possible Labels)", "",
        #                                           "CSV Files (*.csv);;All Files (*)")

        file_path = "Data/ColorsCSV/unique_labels.csv"
        if file_path:
            try:
                df = pd.read_csv(file_path, delimiter=';', header=None)
                df.columns = range(df.shape[1])
                self.dataframes[2] = df
                self.df_possible_labels = df
                self.import_button3.setText(f"Possible Labels File: {file_path.split('/')[-1]} (Loaded) ✔")
            except Exception as e:
                self.imported_files_label.setText(f"Possible Labels File: Error loading ({str(e)})")


def main():
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
