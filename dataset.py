import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

def load_csv_files(folder_path):
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                df = pd.read_csv(file_path, header=None, names=['word'], on_bad_lines='skip')  # Skip bad lines
                df['label'] = file_name.replace('.csv', '')  # Use the filename as the label
                all_data.append(df)
            except pd.errors.ParserError as e:
                print(f"Error reading {file_name}: {e}")
    return pd.concat(all_data, ignore_index=True)

def shuffle_and_split_data(df, test_size=0.30, random_state=42):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True) 
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_data, test_data


class CSVDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=150):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = str(self.dataframe.iloc[idx]['word'])
        label = self.dataframe.iloc[idx]['label']
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()  
        attention_mask = encoding['attention_mask'].squeeze()  
        
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)


def create_data_loaders(train_data, test_data, tokenizer, batch_size):
    train_dataset = CSVDataset(train_data, tokenizer)
    test_dataset = CSVDataset(test_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_data(folder_path,tokenizer, batch_size=32):
    df = load_csv_files(folder_path)
    train_data, test_data = shuffle_and_split_data(df)
    train_loader, test_loader = create_data_loaders(train_data, test_data,tokenizer,batch_size=batch_size)
    return train_data,test_data,train_loader, test_loader


