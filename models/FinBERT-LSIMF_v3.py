''' FinBERT-LSIMF with attention, incremental learning, validation windows, anomaly detection, regularization techniques, monitoring systems '''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import ta
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow setup
mlflow.set_experiment("FinBERT-LSIMF-Advanced")

class FinancialDataset(Dataset):
    def __init__(self, price_data, news_data, sequence_length=7):
        self.price_data = price_data
        self.news_data = news_data
        self.sequence_length = sequence_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.price_data) - self.sequence_length

    def __getitem__(self, idx):
        price_sequence = self.price_data[idx:idx+self.sequence_length]
        news_sequence = self.news_data[idx:idx+self.sequence_length]
        
        encoded_news = self.tokenizer(news_sequence, padding=True, truncation=True, max_length=128, return_tensors="pt")
        
        return {
            'price': torch.tensor(price_sequence.values, dtype=torch.float32),
            'news_input_ids': encoded_news['input_ids'].squeeze(),
            'news_attention_mask': encoded_news['attention_mask'].squeeze(),
            'target': torch.tensor(self.price_data.iloc[idx+self.sequence_length]['Close'], dtype=torch.float32)
        }

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)

    def forward(self, hidden_states):
        attention_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        return attention_output

class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)

    def forward(self, lstm_hidden, bert_hidden):
        attention_output, _ = self.attention(lstm_hidden, bert_hidden, bert_hidden)
        return attention_output

class AttentionFinBERTLSIMF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(AttentionFinBERTLSIMF, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.self_attention = SelfAttention(hidden_size)
        self.cross_attention = CrossAttention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, price, news_input_ids, news_attention_mask):
        bert_output = self.bert(input_ids=news_input_ids, attention_mask=news_attention_mask)
        bert_hidden = bert_output.last_hidden_state
        
        lstm_out, _ = self.lstm(price)
        lstm_attended = self.self_attention(lstm_out)
        cross_attended = self.cross_attention(lstm_attended, bert_hidden)
        
        combined = torch.cat((lstm_attended[:, -1, :], cross_attended[:, -1, :]), dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)
        return output.squeeze()

def prepare_data(symbol, start_date, end_date):
    price_data = yf.download(symbol, start=start_date, end=end_date)
    
    # Calculate technical indicators
    price_data['RSI'] = ta.momentum.RSIIndicator(price_data['Close']).rsi()
    price_data['MACD'] = ta.trend.MACD(price_data['Close']).macd()
    price_data['BB_upper'], price_data['BB_middle'], price_data['BB_lower'] = ta.volatility.BollingerBands(price_data['Close']).bollinger_hband(), ta.volatility.BollingerBands(price_data['Close']).bollinger_mavg(), ta.volatility.BollingerBands(price_data['Close']).bollinger_lband()
    
    scaler = MinMaxScaler()
    price_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']] = scaler.fit_transform(price_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']])
    
    # Placeholder for news data
    news_data = pd.DataFrame({'Date': price_data.index, 'News': ['Sample news ' + str(i) for i in range(len(price_data))]})
    news_data.set_index('Date', inplace=True)
    
    return price_data, news_data

def detect_anomalies(data, contamination=0.01):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso_forest.fit_predict(data)
    return anomalies == -1

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            price = batch['price'].to(device)
            news_input_ids = batch['news_input_ids'].to(device)
            news_attention_mask = batch['news_attention_mask'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            output = model(price, news_input_ids, news_attention_mask)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                price = batch['price'].to(device)
                news_input_ids = batch['news_input_ids'].to(device)
                news_attention_