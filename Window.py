import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QStackedWidget, QLabel, QFileDialog, QTextEdit, QComboBox, \
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.combo_boxes = None
        self.stacked_widget = None
        self.dataframes = [None, None, None]
        self.df_possible_labels = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("SmartLabeler")
        self.setGeometry(200, 200, 400, 300)

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

        import_button1 = QPushButton("Import File 1")
        import_button2 = QPushButton("Import File 2")
        import_button3 = QPushButton("Import File 3")

        import_button1.clicked.connect(self.importData)
        import_button2.clicked.connect(self.importLabels)
        import_button3.clicked.connect(self.importLabelTypes)

        self.imported_files_label = QLabel()

        page1.layout = QVBoxLayout()
        page1.layout.addWidget(import_button1)
        page1.layout.addWidget(import_button2)
        page1.layout.addWidget(import_button3)
        page1.layout.addWidget(self.imported_files_label)

        page1.setLayout(page1.layout)
        self.stacked_widget.addWidget(page1)
        self.stacked_widget.addWidget(page2)
        self.stacked_widget.addWidget(page3)

        self.text_edit = QTextEdit()
        page2.layout = QVBoxLayout()
        page2.layout.addWidget(self.text_edit)
        page2.setLayout(page2.layout)

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

        scroll_area = QScrollArea(self)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        df = pd.concat(self.dataframes[:2], axis=1)
        nan_mask = df.isna().any(axis=1)
        df_known = df[~nan_mask]
        df_predict = df[nan_mask]

        X = df_known.iloc[:, :-1]
        y = df_known.iloc[:, -1]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)

        predictions = model.predict(df_predict.iloc[:100, :-1])

        self.combo_boxes = []

        for i, prediction in enumerate(predictions):
            h_layout = QHBoxLayout()

            value_label = QLabel(f"Value {i}: {df_predict.iloc[i, 0]}")
            combo_box = QComboBox()
            combo_box.addItems(self.df_possible_labels.iloc[:, 0])
            combo_box.setCurrentText(str(prediction))

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
        for i, combo_box in enumerate(self.combo_boxes):
            corrected_label = combo_box.currentText()

    def importData(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File 1 (Data)", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                df = pd.read_csv(file_path, delimiter=';', header=None)
                df.columns = range(df.shape[1])
                self.dataframes[0] = df
                self.imported_files_label.setText(f"Data File: {file_path} (Loaded)")
            except Exception as e:
                self.imported_files_label.setText(f"Data File: Error loading ({str(e)})")

    def importLabels(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File 2 (Class Labels)", "",
                                                   "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                df = pd.read_csv(file_path, delimiter=';', header=None)
                df.columns = range(df.shape[1])
                self.dataframes[1] = df
                self.imported_files_label.setText(f"Class Labels File: {file_path} (Loaded)")
            except Exception as e:
                self.imported_files_label.setText(f"Class Labels File: Error loading ({str(e)})")

    def importLabelTypes(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File 3 (Possible Labels)", "",
                                                   "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                df = pd.read_csv(file_path, delimiter=';', header=None)
                df.columns = range(df.shape[1])
                self.dataframes[2] = df
                self.df_possible_labels = df
                self.imported_files_label.setText(f"Possible Labels File: {file_path} (Loaded)")
            except Exception as e:
                self.imported_files_label.setText(f"Possible Labels File: Error loading ({str(e)})")


def main():
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
