
# Craft Assist Project

This project involves the use of various machine learning and data processing libraries to analyze and manipulate data related to craft assistance.

## Table of Contents

- [Usage](#usage)
- [Files](#files)
- [Requirements](#requirements)
- [License](#license)

## Installation

1.**Install the required dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

2.**Install additional dependencies for PyTorch:**

   ```sh
   pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

1.**Run the script:**

   ```sh
   python Window.py
   ```

## Files

- `Window.py`: Main script file for the project.
- `craft_assist_uniqe_labels.csv`: Contains unique labels for the craft assistance data.
- `craft_assist_all_features.csv`: Contains all features related to the craft assistance data.
- `craft_assist_labels.csv`: Contains labels for the craft assistance data.
- `requirements.txt`: Lists the dependencies required for the project.

## Requirements

- Python 3.10
- Libraries listed in `requirements.txt`
- Additional PyTorch packages as specified in the installation section
