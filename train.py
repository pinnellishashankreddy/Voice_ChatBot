import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'intents.json')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'chatbot_model.keras')
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')

def load_dataset():
    """Load intents dataset from JSON file."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = []
    labels = []
    
    for intent_data in data.get('intents', []):
        intent = intent_data['intent']
        for example in intent_data['examples']:
            sentences.append(example.strip())
            labels.append(intent)
            
    print(f"Loaded {len(sentences)} training samples across {len(set(labels))} intents.")
    return sentences, labels

def train_model():
    print("=" * 60)
    print("VOCAMIND - TRAINING DEEP LEARNING INTENT CLASSIFIER (BiLSTM)")
    print("=" * 60)

    # 1. Load Dataset
    sentences, labels = load_dataset()

    # 2. Tokenization & Preprocessing
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>", lower=True)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    
    sequences = tokenizer.texts_to_sequences(sentences)
    max_len = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    # 3. Label Encoding
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Max Sequence Length: {max_len}")
    print(f"Number of Intent Classes: {num_classes}")

    # 4. Build Deep Learning Model (BiLSTM Architecture)
    # Input -> Tokenizer -> Padding -> Embedding Layer -> Bidirectional LSTM -> Dropout -> Dense -> Softmax
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 5. Train the Model
    history = model.fit(
        padded_sequences, encoded_labels,
        epochs=120,
        batch_size=8,
        verbose=1
    )

    # 6. Save Model Artifacts
    model.save(MODEL_PATH)
    
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)

    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)

    metadata = {
        "vocab_size": vocab_size,
        "max_len": int(max_len),
        "num_classes": num_classes,
        "classes": list(label_encoder.classes_),
        "default_confidence_threshold": 0.60
    }

    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    final_acc = history.history['accuracy'][-1]
    print("\n" + "=" * 60)
    print(f"SUCCESS: Model training completed!")
    print(f"Final Training Accuracy: {final_acc * 100:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Tokenizer saved to: {TOKENIZER_PATH}")
    print(f"Label Encoder saved to: {LABEL_ENCODER_PATH}")
    print("=" * 60)
    return history

if __name__ == '__main__':
    train_model()
