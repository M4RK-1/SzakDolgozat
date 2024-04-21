import sys
import threading
import torch
import numpy as np

from sklearn.metrics import accuracy_score
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QPushButton, QStackedWidget, QLabel, QFileDialog, QTextEdit, QComboBox, \
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QFrame, QMessageBox
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
        self.progress_bar.setValue(0)


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
    prediction_ready = pyqtSignal(list, pd.DataFrame)
    update_training_status = pyqtSignal(str)
    update_accuracy_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()

        self.combo_box_feature = None
        self.scrollLayout = None
        self.scrollWidget = None
        self.scrollArea = None
        self.page3Layout = None
        self.page3 = None
        self.page2 = None
        self.page1 = None
        self.unique_labels_num = None
        self.use_features = None
        self.unique_labels = None
        self.cut_labels = None
        self.all_features = None
        self.buttons_layout = None
        self.fine_tune_model_button = None
        self.text_edit = None
        self.import_button3 = None
        self.import_button2 = None
        self.predict_next_batch_button = None
        self.import_button1 = None
        self.imported_files_label = None
        self.combo_boxes = None
        self.stacked_widget = None
        self.dataframes = [None, None, None]
        self.output = pd.DataFrame()
        self.df_possible_labels = None
        self.progress_bar = None
        self.training_status_label = QLabel()
        self.initUI()
        self.predicted_data = pd.DataFrame(columns=['Features', 'PredictedLabels'])
        self.update_training_status.connect(self.updateTrainingStatusLabel)
        self.update_accuracy_signal.connect(self.updateAccuracyOnButton)

    def updateTrainingStatusLabel(self, text):
        if text == "HIDE_LABEL":
            self.training_status_label.hide()
        else:
            self.training_status_label.setText(text)
            self.training_status_label.show()

    def updateAccuracyOnButton(self, accuracy):
        self.fine_tune_model_button.setText(f"Fine-tune Model (Accuracy: {accuracy:.2f}%)")

    def initUI(self):
        self.setWindowTitle("SmartLabeler")
        self.setGeometry(200, 200, 1400, 700)

        layout = QVBoxLayout()
        self.stacked_widget = QStackedWidget()
        self.page1 = QWidget()
        self.page2 = QWidget()
        self.page3 = QWidget()
        self.page3Layout = QVBoxLayout(self.page3)

        self.setupPage3()

        button1 = QPushButton("Load Data")
        button1.clicked.connect(self.LoadFilesPage)
        layout.addWidget(button1)

        button_layout = QHBoxLayout()

        button2 = QPushButton("Check Data")
        button2.clicked.connect(self.SeeLoadedPage)
        button_layout.addWidget(button2, 1)

        button4 = QPushButton("Save Results")
        button4.clicked.connect(self.save_results)
        button_layout.addWidget(button4, 1)

        layout.addLayout(button_layout)

        button3 = QPushButton("Predict")
        button3.clicked.connect(self.CalculationsPage)
        layout.addWidget(button3)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)

        self.import_button1 = QPushButton("Load Data File")
        self.import_button2 = QPushButton("Load Known Labels File")
        self.import_button3 = QPushButton("Load Unique Labels File")

        self.import_button1.clicked.connect(self.importData)
        self.import_button2.clicked.connect(self.importLabels)
        self.import_button3.clicked.connect(self.importLabelTypes)

        self.imported_files_label = QLabel()

        self.page1.layout = QVBoxLayout()
        self.page1.layout.addWidget(self.import_button1)
        self.page1.layout.addWidget(self.import_button2)
        self.page1.layout.addWidget(self.import_button3)
        self.page1.layout.addWidget(self.imported_files_label)

        self.page1.setLayout(self.page1.layout)
        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.page3)

        self.text_edit = QTextEdit()
        self.page2.layout = QVBoxLayout()
        self.page2.layout.addWidget(self.text_edit)
        self.page2.setLayout(self.page2.layout)

        self.progress_bar = QProgressBar(self)

        layout.addWidget(self.training_status_label)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def setupPage3(self):
        self.scrollArea = QScrollArea(self.page3)
        self.scrollWidget = QWidget()
        self.scrollLayout = QVBoxLayout(self.scrollWidget)
        self.scrollArea.setWidget(self.scrollWidget)
        self.scrollArea.setWidgetResizable(True)
        self.page3Layout.addWidget(self.scrollArea)

        self.predict_next_batch_button = QPushButton("Predict Next Batch", self)
        self.predict_next_batch_button.clicked.connect(self.predict_for_next_batch)

        self.fine_tune_model_button = QPushButton("Fine-tune Model", self)
        self.fine_tune_model_button.clicked.connect(self.finetune_model)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.predict_next_batch_button)
        self.buttons_layout.addWidget(self.fine_tune_model_button)
        self.page3Layout.addLayout(self.buttons_layout)

    def LoadFilesPage(self):
        self.stacked_widget.setCurrentIndex(0)

    def SeeLoadedPage(self):
        self.stacked_widget.setCurrentIndex(1)
        concatenated_df = pd.concat(self.dataframes[:2], axis=1)
        text_to_display = concatenated_df.iloc[1:].to_string(index=False)
        self.text_edit.setPlainText(text_to_display)

    def PredictWithModel(self, additional_features):
        self.update_training_status.emit("Prediction in progress...")

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_bert")

        total_predictions = len(additional_features)
        predicted_classes = []

        for i, row in enumerate(additional_features.iterrows()):
            _, data = row
            text = data[0]
            inputs = tokenizer(text, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            predicted_probs = outputs.logits.softmax(dim=-1)
            predicted_class = predicted_probs.argmax().item()
            predicted_classes.append(predicted_class)

            progress = int((i + 1) / total_predictions * 100)
            self.progress_bar.setValue(progress)

        self.progress_bar.setValue(100)

        self.update_training_status.emit("HIDE_LABEL")

        return predicted_classes

    def CalculationsPage(self):
        self.stacked_widget.setCurrentIndex(2)
        self.page3 = self.stacked_widget.widget(2)

        def train_model(features, labels):
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

            cut_labels_numer = labels.iloc[:, 0].apply(lambda x: self.unique_labels_num.index(x))

            cut_data = pd.concat([features, cut_labels_numer], axis=1)

            train_data, eval_data = train_test_split(cut_data, test_size=0.2)

            train_encodings = tokenizer(train_data.iloc[:, 0].tolist(), truncation=True, padding='max_length',
                                        max_length=512)
            eval_encodings = tokenizer(eval_data.iloc[:, 0].tolist(), truncation=True, padding='max_length',
                                       max_length=512)

            train_labels = train_data.iloc[:, 1].tolist()
            eval_labels = eval_data.iloc[:, 1].tolist()

            train_dataset = CustomDataset(train_encodings, train_labels)
            eval_dataset = CustomDataset(eval_encodings, eval_labels)

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                       num_labels=len(self.unique_labels_num))

            training_args = TrainingArguments(
                output_dir="./results",
                learning_rate=2e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                num_train_epochs=15,
                weight_decay=0.01
            )

            num_update_steps = ((len(train_dataset) // training_args.per_device_train_batch_size + 1)
                                * training_args.num_train_epochs)

            progress_callback = ProgressCallback(self.progress_bar, num_update_steps, self.update_training_status)

            def compute_metrics(eval_pred):
                acc_logits, acc_labels = eval_pred
                acc_predictions = np.argmax(acc_logits, axis=-1)
                return {"accuracy": accuracy_score(acc_labels, acc_predictions)}

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[progress_callback],
            )

            trainer.train()
            eval_results = trainer.evaluate()
            accuracy = eval_results.get('eval_accuracy', 0) * 100
            self.update_accuracy_signal.emit(accuracy)

            trainer.save_model("fine_tuned_bert")

            self.use_features, additional_features = self.use_features[20:], self.use_features[:20]

            self.update_predict_button_text()

            predicted_labels = [self.unique_labels_num[prediction] for prediction in
                                self.PredictWithModel(additional_features)]

            # Emit the signal after training is finished
            self.prediction_ready.emit(predicted_labels, additional_features)

        # Connect the signal to a slot that updates the UI
        self.prediction_ready.connect(self.display_predicted_labels)

        self.use_features, features = (self.use_features[self.cut_labels.shape[0]:],
                                       self.use_features[:self.cut_labels.shape[0]])

        self.update_predict_button_text()

        # Start training thread
        train_thread = threading.Thread(target=train_model(features, self.cut_labels))
        train_thread.start()



    def update_predict_button_text(self):
        remaining_items = len(self.use_features)
        self.predict_next_batch_button.setText(f"Predict Next Batch (remaining: {remaining_items})")

    def display_predicted_labels(self, predicted_labels, additional_features):
        # Clear the existing widgets in the scroll layout
        self.clearLayout(self.scrollLayout)
        self.combo_boxes = []
        self.combo_box_feature = []

        for i, prediction in enumerate(predicted_labels):
            if i > 0:
                separator = QFrame()
                separator.setFrameShape(QFrame.HLine)
                separator.setFrameShadow(QFrame.Sunken)
                self.scrollLayout.addWidget(separator)

            h_layout = QHBoxLayout()

            value_label = QLabel(f"Value {i}: {additional_features.iloc[i, 0]}")
            value_label.setWordWrap(True)
            value_label.setMaximumWidth(700)
            self.combo_box_feature.append(additional_features.iloc[i, 0])
            combo_box = QComboBox()
            combo_box.addItems(self.df_possible_labels.iloc[:, 0])
            combo_box.setCurrentText(str(prediction))
            combo_box.setMaximumWidth(150)
            self.combo_boxes.append(combo_box)

            h_layout.addWidget(value_label)
            h_layout.addWidget(combo_box)

            self.scrollLayout.addLayout(h_layout)

    def clearLayout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                self.clearLayout(item.layout())

    def predict_for_next_batch(self):
        corrected_labels = []

        for combo_box in self.combo_boxes:
            corrected_label = combo_box.currentText()
            corrected_labels.append(corrected_label)

        corrected_labels_df = pd.DataFrame(corrected_labels, columns=['Labels'])

        nan_indices = self.output[self.output['Labels'].isna()].index
        update_indices = nan_indices[:len(corrected_labels_df)]
        self.output.loc[update_indices, 'Labels'] = corrected_labels_df['Labels'].values[:len(update_indices)]

        self.use_features, additional_features = self.use_features[20:], self.use_features[:20]

        self.update_predict_button_text()

        predicted_labels = [self.unique_labels_num[prediction] for prediction in
                            self.PredictWithModel(additional_features)]

        self.display_predicted_labels(predicted_labels, additional_features)

        print("Next batch done")



    def finetune_model(self):
        corrected_labels = []
        corrected_features = self.combo_box_feature

        for combo_box in self.combo_boxes:
            corrected_label = combo_box.currentText()
            corrected_labels.append(corrected_label)

        corrected_labels = list(map(self.unique_labels_num.index, corrected_labels))

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        encodings = tokenizer(corrected_features, truncation=True, padding=True, max_length=512)

        full_dataset = CustomDataset(encodings, corrected_labels)

        train_size = int(0.8 * len(full_dataset))
        train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset,
                                                                    [train_size, len(full_dataset) - train_size])

        model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_bert")

        training_args = TrainingArguments(
            output_dir="./results_finetune",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=15,
            #num_train_epochs=1,
            weight_decay=0.01
        )

        num_update_steps = ((len(train_dataset) // training_args.per_device_train_batch_size + 1)
                            * training_args.num_train_epochs)

        progress_callback = ProgressCallback(self.progress_bar, num_update_steps, self.update_training_status)

        def compute_metrics(eval_pred):
            acc_logits, acc_labels = eval_pred
            acc_predictions = np.argmax(acc_logits, axis=-1)
            return {"accuracy": accuracy_score(acc_labels, acc_predictions)}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[progress_callback],
        )

        trainer.train()
        eval_results = trainer.evaluate()
        accuracy = eval_results.get('eval_accuracy', 0) * 100
        self.update_accuracy_signal.emit(accuracy)

        trainer.save_model("fine_tuned_bert")

    def importData(self):
        # uncomment this line to use the file dialog
        # file_path, _ = QFileDialog.getOpenFileName(self, "Select File 1 (Data)", "",
        #                                           "CSV Files (*.csv);;All Files (*)")

        file_path = "Data/CraftAssistCSV/craft_assist_all_features.csv"
        #file_path = "Data/ColorsCSV/features.csv"
        if file_path:
            try:
                df = pd.read_csv(file_path, delimiter=';', header=None)
                df.columns = range(df.shape[1])
                self.dataframes[0] = df
                self.all_features = self.dataframes[0]
                self.output['Features'] = self.all_features
                self.use_features = self.all_features
                self.update_predict_button_text()
                self.import_button1.setText(f"Data File: {file_path.split('/')[-1]} (Loaded) ✔")
            except Exception as e:
                self.imported_files_label.setText(f"Data File: Error loading ({str(e)})")

    def importLabels(self):
        # uncomment this line to use the file dialog
        # file_path, _ = QFileDialog.getOpenFileName(self, "Select File 2 (Class Labels)", "",
        #                                           "CSV Files (*.csv);;All Files (*)")

        file_path = "Data/CraftAssistCSV/craft_assist_labels.csv"
        #file_path = "Data/ColorsCSV/cut_labels.csv"
        if file_path:
            try:
                df = pd.read_csv(file_path, delimiter=';', header=None)
                df.columns = range(df.shape[1])
                self.dataframes[1] = df
                self.cut_labels = self.dataframes[1]

                self.output['Labels'] = self.cut_labels
                self.import_button2.setText(f"Class Labels File: {file_path.split('/')[-1]} (Loaded) ✔")
            except Exception as e:
                self.imported_files_label.setText(f"Class Labels File: Error loading ({str(e)})")

    def importLabelTypes(self):
        # uncomment this line to use the file dialog
        # file_path, _ = QFileDialog.getOpenFileName(self, "Select File 3 (Possible Labels)", "",
        #                                           "CSV Files (*.csv);;All Files (*)")

        file_path = "Data/CraftAssistCSV/craft_assist_uniqe_labels.csv"
        #file_path = "Data/ColorsCSV/unique_labels.csv"
        if file_path:
            try:
                df = pd.read_csv(file_path, delimiter=';', header=None)
                df.columns = range(df.shape[1])
                self.dataframes[2] = df
                self.unique_labels = self.dataframes[2]
                self.df_possible_labels = df
                self.unique_labels_num = self.cut_labels.iloc[:, 0].unique().tolist()
                self.import_button3.setText(f"Possible Labels File: {file_path.split('/')[-1]} (Loaded) ✔")
            except Exception as e:
                self.imported_files_label.setText(f"Possible Labels File: Error loading ({str(e)})")

    def save_results(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "",
                                                   "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            if not file_name.endswith('.csv'):
                file_name += '.csv'
            self.output.to_csv(file_name, index=False)
            QMessageBox.information(self, "Save File", "Results saved successfully!")


def main():
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
