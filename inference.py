import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from main import get_model

idx_to_class = {
    0: "Noun",
    1: "Verb",
    2: "Adjective",
    3: "Adverb",
    4: "Pronoun",
    5: "Conjunction"
}

def load_model(model_path, device):
    model = get_model()  
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.to(device)
    model.eval()
    return model

def preprocess(text, tokenizer, max_length=128):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoding['input_ids'], encoding['attention_mask']

def infer(model, tokenizer, text, device):
    input_ids, attention_mask = preprocess(text, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits  
        else:
            logits = outputs[0]  
        
        predictions = torch.argmax(logits, dim=-1)  

    predicted_class_name = idx_to_class.get(predictions.item(), "Unknown")
    
    return predicted_class_name

if __name__ == "__main__":
    model_path = '/kaggle/working/test/Finetuning_bert_POS/checkpoint.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = load_model(model_path, device)
    
    text = "Good"
    predicted_class = infer(model, tokenizer, text, device)
    print(f'Predicted class: {predicted_class}')
