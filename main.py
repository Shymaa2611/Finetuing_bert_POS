from training import train
from validation import evaluate
import torch
from transformers import BertModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import get_data
from transformers import BertTokenizer
from model import BertPOS
def get_model():
   model=BertPOS(6)
   return model
   
def run(model,train_set,test_set,train_loader,test_loader,optimizer, criterion):
    num_epochs=10
    for epoch in range(num_epochs):
      train(model, optimizer, train_set,train_loader, criterion)
      evaluate(model,test_set,test_loader, criterion)
    torch.save(model.state_dict(), ' checkpoint.pt')

if __name__=="__main__":
   data_path="data"
   model=get_model() 
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   train_set,test_set,train_loader,test_loader=get_data(data_path,tokenizer)
   criterion = nn.CrossEntropyLoss()  
   optimizer =optim.AdamW(model.parameters(), lr=2e-5)
   run(model,train_set,test_set,train_loader,test_loader,optimizer,criterion)