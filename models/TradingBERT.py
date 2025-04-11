import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class TradingBERT(nn.Module):
    def __init__(self, num_classes, seq_length):
        super(TradingBERT, self).__init__()
        self.config = BertConfig(
            vocab_size=30522,  # Standard BERT vocab size
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=seq_length
        )
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Usage
model = TradingBERT(num_classes=3, seq_length=128)  # 3 classes for buy, sell, hold, and sequence length of 128