from training import train
from evaluate import evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data
from transformers import BertTokenizer
from model import BertPOS
def get_model():
   model=BertPOS(6)
   return model
   
def run(model, train_set, test_set, train_loader, test_loader, optimizer, criterion, device):
    num_epochs = 3
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train(model, optimizer, train_set, train_loader, criterion, device)
        evaluate(model, test_set, test_loader, criterion, device)
    
    torch.save(model.state_dict(), 'checkpoint.pt')

if __name__ == "__main__":
    data_path = "data"
    model = get_model() 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_set, test_set, train_loader, test_loader = get_data(data_path, tokenizer)
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    run(model, train_set, test_set, train_loader, test_loader, optimizer, criterion, device)
