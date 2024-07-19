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
        
        # Tokenize news
        encoded_news = self.tokenizer(news_sequence, padding=True, truncation=True, return_tensors="pt")
        
        return {
            'price': torch.tensor(price_sequence.values, dtype=torch.float32),
            'news_input_ids': encoded_news['input_ids'].squeeze(),
            'news_attention_mask': encoded_news['attention_mask'].squeeze(),
            'target': torch.tensor(self.price_data.iloc[idx+self.sequence_length]['Close'], dtype=torch.float32)
        }

class FinBERTLSIMF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(FinBERTLSIMF, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size + 768, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, price, news_input_ids, news_attention_mask):
        # Process news data through BERT
        bert_output = self.bert(input_ids=news_input_ids, attention_mask=news_attention_mask)
        bert_embeddings = bert_output.last_hidden_state[:, 0, :]  # Use [CLS] token embedding
        
        # Combine price data and BERT embeddings
        combined_input = torch.cat((price, bert_embeddings.unsqueeze(1).repeat(1, price.shape[1], 1)), dim=2)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(combined_input)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        
        # Final prediction
        output = self.fc(lstm_out)
        return output.squeeze()

def prepare_data(symbol, start_date, end_date):
    # Fetch price data
    price_data = yf.download(symbol, start=start_date, end=end_date)
    
    # Calculate technical indicators
    price_data['RSI'] = ta.momentum.RSIIndicator(price_data['Close']).rsi()
    price_data['MACD'] = ta.trend.MACD(price_data['Close']).macd()
    price_data['BB_upper'], price_data['BB_middle'], price_data['BB_lower'] = ta.volatility.BollingerBands(price_data['Close']).bollinger_hband(), ta.volatility.BollingerBands(price_data['Close']).bollinger_mavg(), ta.volatility.BollingerBands(price_data['Close']).bollinger_lband()
    
    # Normalize data
    scaler = MinMaxScaler()
    price_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']] = scaler.fit_transform(price_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower']])
    
    # Fetch news data (this is a placeholder - you need to implement actual news fetching)
    news_data = pd.DataFrame({'Date': price_data.index, 'News': ['Sample news ' + str(i) for i in range(len(price_data))]})
    news_data.set_index('Date', inplace=True)
    
    return price_data, news_data

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                price = batch['price'].to(device)
                news_input_ids = batch['news_input_ids'].to(device)
                news_attention_mask = batch['news_attention_mask'].to(device)
                target = batch['target'].to(device)
                
                output = model(price, news_input_ids, news_attention_mask)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

def main():
    # Prepare data
    price_data, news_data = prepare_data('EURUSD=X', '2020-01-01', '2023-01-01')
    
    # Create datasets
    dataset = FinancialDataset(price_data, news_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    input_size = price_data.shape[1]  # Number of features
    hidden_size = 128
    num_layers = 2
    model = FinBERTLSIMF(input_size, hidden_size, num_layers)
    
    # Train model
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()