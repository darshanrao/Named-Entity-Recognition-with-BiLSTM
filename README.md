# Named Entity Recognition with BiLSTM

This project implements a Named Entity Recognition (NER) system using a Bidirectional Long Short-Term Memory (BiLSTM) neural network. The system is designed to read in a dataset, train a BiLSTM model on this data, and then use the trained model to predict entity tags for unseen text.

## Prerequisites

Before you run the code, ensure that you have the following prerequisites installed:

- Python 3.x
- PyTorch (1.x or later)
- Torchtext
- NumPy
- scikit-learn

You can install these packages using `pip`:

```bash
pip install torch torchtext numpy sklearn


## Dataset Format

The code expects the data to be in a specific format where each line contains a word followed by its corresponding entity tag, separated by a space. Sentences in the dataset should be separated by blank lines. Here is an example:

Word1 Tag1
Word2 Tag2
Word3 Tag3

Word1 Tag1
Word2 Tag2



## File Structure

- `data/train`: Training data file.
- `data/dev`: Development (or validation) data file.
- `data/test`: Test data file, containing only words without tags.

## Running the Code
### TASK 1
### Data Preparation

Place your train, dev, and test datasets in the `data` folder. Ensure they are named `train`, `dev`, and `test` respectively.

### Training the Model

Run the provided script to train the BiLSTM model. This process involves reading the data, creating a vocabulary, converting the data to indices, and then training the model using the training data.

```bash
python task1.py


### Model Evaluation

After training, the script will automatically evaluate the model on the development dataset and output the performance metrics.

### Predicting Tags

The trained model will be used to predict tags for the test dataset. The predictions will be saved to a file named `test1.out`.

## Output Files

- `dev1.out`: Contains the predictions of the development dataset.
- `test1.out`: Contains the predictions of the test dataset.

Both files will have the same format as the input data, but with predicted tags replacing the actual tags.

## Notes

- The model's state is saved to `blstm1.pt`; ensure that this file is in the same directory as your script when running predictions.
- Modify the batch size, learning rate, or model architecture in the script as needed to improve performance.


## TASK 2
### Data Preparation

Place your train, dev, and test datasets in the `data` folder. Ensure they are named `train`, `dev`, and `test` respectively.

### Training the Model

Run the provided script to train the BiLSTM model. This process involves reading the data, creating a vocabulary, converting the data to indices, and then training the model using the training data.

```bash
python task2.py


### Model Evaluation

After training, the script will automatically evaluate the model on the development dataset and output the performance metrics.

### Predicting Tags

The trained model will be used to predict tags for the test dataset. The predictions will be saved to a file named `test2.out`.

## Output Files

- `dev2.out`: Contains the predictions of the development dataset.
- `test2.out`: Contains the predictions of the test dataset.

Both files will have the same format as the input data, but with predicted tags replacing the actual tags.

## Notes

- The model's state is saved to `blstm2.pt`; ensure that this file is in the same directory as your script when running predictions.
- Modify the batch size, learning rate, or model architecture in the script as needed to improve performance.





