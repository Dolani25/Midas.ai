import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Attention, GlobalAveragePooling1D, Reshape, Dropout
from tensorflow.keras.models import Model
from transformers import TFRobertaModel, RobertaTokenizer

def create_trabsa_model(max_length=128, num_classes=3):
    # Input layers
    input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

    # Load pre-trained RoBERTa model
    roberta_model = TFRobertaModel.from_pretrained("roberta-base")
    
    # Get RoBERTa embeddings
    roberta_outputs = roberta_model(input_ids, attention_mask=attention_mask)[0]
    
    # Global Average Pooling
    gap_output = GlobalAveragePooling1D()(roberta_outputs)
    
    # Reshape
    reshaped = Reshape((1, -1))(gap_output)
    
    # Bidirectional LSTM
    bilstm = Bidirectional(LSTM(512, return_sequences=True))(reshaped)
    
    # Attention mechanism
    attention = Attention()([bilstm, bilstm])
    
    # Dense layers with ReLU and Dropout
    dense1 = Dense(512, activation="relu")(attention)
    dropout1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(256, activation="relu")(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    
    dense3 = Dense(128, activation="relu")(dropout2)
    
    # Output layer
    output = Dense(num_classes, activation="softmax")(dense3)
    
    # Create model
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    
    return model

# Function to tokenize text
def tokenize_text(texts, tokenizer, max_length):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="tf")

# Main execution
if __name__ == "__main__":
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = create_trabsa_model()
    
    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Print model summary
    model.summary()
    
    # Example usage (you would replace this with your actual data loading and preprocessing)
    example_texts = ["This is a positive tweet!", "This is a negative tweet.", "This is a neutral statement."]
    tokenized = tokenize_text(example_texts, tokenizer, max_length=128)
    
    # Make predictions (this is just a demonstration, not trained model)
    predictions = model.predict([tokenized["input_ids"], tokenized["attention_mask"]])
    print("Example predictions shape:", predictions.shape)