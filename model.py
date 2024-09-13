from transformers import BertModel, BitsAndBytesConfig
import torch.nn as nn
import torch

class BertPOS(nn.Module):
    def __init__(self, num_labels=6):
        super(BertPOS, self).__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.bert = BertModel.from_pretrained('bert-base-uncased', quantization_config=bnb_config)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = pooled_output.to(self.classifier.weight.dtype)

        logits = self.classifier(pooled_output)
        return logits
