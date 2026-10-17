import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'intents.json')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'chatbot_model.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'model', 'tokenizer.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'model', 'label_encoder.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model', 'model_metadata.json')

def evaluate_model():
    print("=" * 60)
    print("VOCAMIND - MODEL EVALUATION SUITE")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print("Model file not found! Please run python train.py first.")
        return None

    # Load artifacts
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)

    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    max_len = metadata['max_len']

    # Load dataset
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    sentences = []
    y_true_labels = []

    for intent_data in dataset.get('intents', []):
        intent = intent_data['intent']
        for example in intent_data['examples']:
            sentences.append(example.strip())
            y_true_labels.append(intent)

    # Convert to sequences & pad
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    y_true = label_encoder.transform(y_true_labels)

    # Predict
    predictions = model.predict(padded_sequences, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"\nEvaluation Results on Complete Dataset ({len(sentences)} samples):")
    print(f"--------------------------------------------------")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"--------------------------------------------------\n")

    print("Detailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix Shape:", cm.shape)

    metrics_result = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "total_samples": len(sentences),
        "intents_evaluated": len(label_encoder.classes_)
    }
    
    return metrics_result

if __name__ == '__main__':
    evaluate_model()
