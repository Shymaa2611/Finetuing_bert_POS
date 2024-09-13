from transformers import BertModel
import torch.nn as nn


class BertPOS(nn.Module):
    def __init__(self, num_labels=6):
        super(BertPOS, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_labels)  

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  
        logits = self.classifier(pooled_output)  
        return logits
