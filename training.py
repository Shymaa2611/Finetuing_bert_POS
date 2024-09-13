import torch

def train(model, optimizer, train_set, train_loader, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        total_acc += (predictions == labels).sum().item()
    
    print(f'Training loss: {total_loss/len(train_loader)} Train acc: {total_acc/len(train_set)*100}%')
