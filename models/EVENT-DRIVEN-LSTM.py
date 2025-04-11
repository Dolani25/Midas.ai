import torch
import torch.nn as nn

class EventDrivenLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(EventDrivenLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.event_embedding = nn.Embedding(num_embeddings=10, embedding_dim=hidden_size)  # Assuming 10 event types
        
    def forward(self, x, events):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        event_embed = self.event_embedding(events)
        out = out + event_embed
        out = self.fc(out[:, -1, :])
        return out

# Usage
model = EventDrivenLSTM(input_size=10, hidden_size=64, num_layers=2, output_size=1)