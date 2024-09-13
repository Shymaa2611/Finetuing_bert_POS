# BERT Fine-Tuning for Part-of-Speech Tagging
- This project demonstrates how to fine-tune the BERT model for the task of Part-of-Speech (POS) 
  tagging. The dataset consists of multiple CSV files,containing words . We fine-tune the pre-trained BERT model and adapt it to handle this specific task.
## Project Structure
graphql

.
├── dataset/             
│   ├── file1.csv       
│   ├── file2.csv
│   └── ...
├── model.py              
├── training.py           
├── evaluate.py                    
├── requirements.txt      
└── README.md            

## Dataset
The dataset is stored in the data/ folder and consists of multiple CSV files. Each CSV file has the following format:

    Each row contains a single word from a sentence.
    The corresponding POS label for each word is inferred from the file name.

    The file name (e.g., verbs.csv, nouns.csv) will be treated as the label for all words in that file.

## Model

We use the BERT model from the Hugging Face Transformers library. The model is fine-tuned to classify each word into one of the POS categories (e.g., noun, verb, adjective). The output layer is a linear classifier on top of BERT's pooled output.

Key Model Components:

    BERT Pretrained Model: bert-base-uncased
    Classifier: Linear layer that maps the BERT output to POS labels.

Setup Instructions
1. Clone the Repository

bash

git clone https://github.com/Shymaa2611/Finetuning_bert_POS.git
cd Finetuning_bert_POS

2. Install Dependencies

Ensure that you have Python 3.8 or higher. Install the required dependencies using:

bash

pip install -r requirements.txt

Requirements:

    torch
    transformers
    sklearn
    pandas

3. Prepare the Dataset

Ensure your dataset folder (datat/) contains CSV files where each file represents a word list for a specific POS category. For example, one file could represent verbs, and another could represent nouns.
4. Training

To start training the model, run the following command:

bash

python training.py --data_path ./dataset --epochs 3 --batch_size 32 --learning_rate 2e-5

Arguments:

    --data_path: Path to the dataset folder containing CSV files.
    --epochs: Number of epochs for training.
    --batch_size: Batch size for training.
    --learning_rate: Learning rate for the optimizer.

5. Evaluation

After training, evaluate the model on a test set using:

bash

python evaluate.py --data_path ./dataset --batch_size 32

6. Inference

To perform inference and predict POS tags for new sentences, use the following script:

bash

python inference.py --sentence "I am running fast"

This will output the predicted POS tags for each word in the sentence.
Code Overview
Model Definition (model.py)

Defines the BertPOS class, which loads a pre-trained BERT model and adds a linear classifier on top.
Data Loading (utils.py)

    load_csv_files(): Reads CSV files from the dataset folder and loads them into a pandas DataFrame.
    CSVDataset class: Converts the DataFrame into a PyTorch Dataset, tokenizes the text, and prepares input tensors for the BERT model.

Training (training.py)

Handles the training loop, optimizer setup, and model saving. Tracks the training loss and accuracy.
Evaluation (evaluate.py)

Loads the trained model and evaluates it on a test set, printing the test accuracy and loss.
Hyperparameter Tuning

You can experiment with the following hyperparameters for better results:

    Learning rate: Adjust between 1e-5 and 5e-5.
    Batch size: Depending on the available GPU memory.
    Max sequence length: Set a limit to the number of tokens processed by BERT.

## Acknowledgements

    The BERT model is from the Hugging Face Transformers library.
    Part-of-Speech tagging is a core NLP task with many applications, such as syntactic parsing and word sense disambiguation.