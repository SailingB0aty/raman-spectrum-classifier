# Raman Classifier #
Python project to classify Raman spectra.

## Data ##
Data has been ommited from this repository due to me being unsure about the liscencing. If you want to run this youll have to provide your own raman spectra in CSV files in the format of each spectrum being one column of 1024 entries.

The repository expects the following file structure:
  data/ - all spectra CSV files
  model/ - saved model outputs
  Test data/ - for testing the models on specific data (CSV files)

## Linux Setup ##
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
'''

## Running ##
Training:
  python3 main.py

Test model on specific dataset:
  python3 test.py
