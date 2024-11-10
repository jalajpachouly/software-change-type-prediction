# -*- coding: utf-8 -*-
"""cd_final.py

This script performs multiclass text classification on software defect reports.
It uses SMOTE to balance the dataset and compares the performance of various
machine learning models on both balanced and unbalanced data.

It generates the following plots:
1. Original Class Distribution
2. Training Data Before SMOTE (Unbalanced)
3. Training Data After SMOTE (Balanced)
4. BERT Training Data Before Oversampling (Unbalanced)
5. BERT Training Data After Oversampling (Balanced)
6. Evaluation of Metrics with all models, box plot - Balanced
7. Evaluation of Metrics with all models, box plot - Unbalanced
8. Evaluation of Metrics with all models, bar plot - Balanced
9. Evaluation of Metrics with all models, bar plot - Unbalanced
10. Word Clouds for each class
"""

# Install necessary libraries
!pip install transformers -q
!pip install datasets -q
!pip install wordcloud -q
!pip install imbalanced-learn -q  # For SMOTE

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
import torch

# Transformers and tokenizers
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Data handling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Machine Learning models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Handling class imbalance
from imblearn.over_sampling import SMOTE

# Handle warnings
import warnings
warnings.filterwarnings('ignore')

# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# For progress bar
from tqdm.notebook import tqdm

# Load the dataset
data = pd.read_csv('dataset.csv')

# Display first few rows
data.head()

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Distribution of labels
print("\nClass distribution:")
print(data['target'].value_counts())

# Plot label distribution
plt.figure(figsize=(10,6))
sns.countplot(x='target', data=data, order=data['target'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Original Class Distribution')
plt.tight_layout()
plt.savefig('original_class_distribution.png')
plt.show()

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Join back into text
    return ' '.join(words)

# Apply preprocessing
tqdm.pandas()
data['cleaned_text'] = data['report'].progress_apply(preprocess_text)

# Display cleaned text
data[['report', 'cleaned_text']].head()

# Encode labels
label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['target'])

# Generate word clouds for each class
def generate_wordclouds(data):
    classes = data['target'].unique()
    for cls in classes:
        text = ' '.join(data[data['target'] == cls]['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {cls}')
        plt.tight_layout()
        plt.savefig(f'wordcloud_{cls}.png')
        plt.close()

generate_wordclouds(data)

# Split data for models without balancing
X_train_ml_unbalanced, X_test_ml, y_train_ml_unbalanced, y_test_ml = train_test_split(
    data['cleaned_text'], data['label_encoded'], test_size=0.2, random_state=42, stratify=data['label_encoded'])

# Plot training label distribution before SMOTE (Unbalanced)
plt.figure(figsize=(12, 5))

sns.countplot(x=y_train_ml_unbalanced, palette='viridis')
plt.title('Training Data Before SMOTE (Unbalanced)')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_, rotation=45)
plt.tight_layout()
plt.savefig('training_data_before_smote.png')
plt.show()

# Vectorize text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf_unbalanced = tfidf_vectorizer.fit_transform(X_train_ml_unbalanced)
X_test_tfidf = tfidf_vectorizer.transform(X_test_ml)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_tfidf_balanced, y_train_ml_balanced = smote.fit_resample(X_train_tfidf_unbalanced, y_train_ml_unbalanced)

# Verify the balancing
print("\nAfter SMOTE, label distribution:")
unique, counts = np.unique(y_train_ml_balanced, return_counts=True)
print(dict(zip(label_encoder.inverse_transform(unique), counts)))

# Plot training label distribution after SMOTE (Balanced)
plt.figure(figsize=(12, 5))
sns.countplot(x=y_train_ml_balanced, palette='viridis')
plt.title('Training Data After SMOTE (Balanced)')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_, rotation=45)
plt.tight_layout()
plt.savefig('training_data_after_smote.png')
plt.show()

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVM': LinearSVC(),
    'Multinomial NB': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

# Function to collect evaluation metrics
def collect_metrics(models, X_train, y_train, X_test, y_test, data_desc):
    metrics_dict = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
    for model_name, model in models.items():
        print(f"\nTraining {model_name} on {data_desc} data...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        metrics_dict['Model'].append(model_name)
        metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics_dict['Precision'].append(report['weighted avg']['precision'])
        metrics_dict['Recall'].append(report['weighted avg']['recall'])
        metrics_dict['F1-Score'].append(report['weighted avg']['f1-score'])
        print(f"Accuracy: {metrics_dict['Accuracy'][-1]:.2f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    return pd.DataFrame(metrics_dict)

# Collect metrics for unbalanced data
metrics_unbalanced = collect_metrics(models, X_train_tfidf_unbalanced, y_train_ml_unbalanced, X_test_tfidf, y_test_ml, "unbalanced")

# Collect metrics for balanced data
metrics_balanced = collect_metrics(models, X_train_tfidf_balanced, y_train_ml_balanced, X_test_tfidf, y_test_ml, "balanced")

# Visualization of Evaluation Metrics

import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics_box(metrics_df, title):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=metrics_df, palette='Set3')
    plt.title(title, fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

def plot_metrics_bar(metrics_df, title):
    metrics_df.plot(kind='bar', figsize=(10,6), colormap='Set2')
    plt.title(title, fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.show()

# Assuming you have a DataFrame 'metrics_df' with your evaluation metrics

# Generate plots for unbalanced data
plot_metrics_box(metrics_unbalanced, 'Unbalanced')
plot_metrics_bar(metrics_unbalanced, 'Unbalanced')

# Generate plots for balanced data
plot_metrics_box(metrics_balanced, 'Balanced')
plot_metrics_bar(metrics_balanced, 'Balanced')

# ------- BERT Section -------

# For BERT, we will balance the training data by oversampling the minority classes

# Prepare data for BERT
# Note: SMOTE cannot be directly applied to text data, so for BERT we will balance by oversampling the minority classes

# Combine training data
train_data_bert = pd.DataFrame({
    'cleaned_text': X_train_ml_unbalanced,
    'label': y_train_ml_unbalanced
})

# Plot BERT training label distribution before oversampling
plt.figure(figsize=(12, 5))
sns.countplot(x='label', data=train_data_bert, palette='magma')
plt.title('BERT Training Data Before Oversampling (Unbalanced)')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_, rotation=45)
plt.tight_layout()
plt.savefig('bert_training_data_before_oversampling.png')
plt.show()

# Oversample the minority classes
from sklearn.utils import resample

train_data_list = [train_data_bert[train_data_bert['label'] == i] for i in range(len(label_encoder.classes_))]
max_size = max(len(d) for d in train_data_list)

train_data_balanced = pd.concat([
    resample(d, replace=True, n_samples=max_size, random_state=42) if len(d) < max_size else d
    for d in train_data_list
])

# Shuffle the balanced training data
train_data_balanced = train_data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Plot BERT training label distribution after oversampling (Balanced)
plt.figure(figsize=(12, 5))
sns.countplot(x='label', data=train_data_balanced, palette='magma')
plt.title('BERT Training Data After Oversampling (Balanced)')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_, rotation=45)
plt.tight_layout()
plt.savefig('bert_training_data_after_oversampling.png')
plt.show()

# Tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(texts):
    return tokenizer(
        texts, add_special_tokens=True, max_length=128, padding='max_length',
        truncation=True, return_attention_mask=True, return_tensors='pt')

# Original unbalanced data
X_train_unbalanced = X_train_ml_unbalanced
y_train_unbalanced = y_train_ml_unbalanced

# Balanced data for BERT
X_train_balanced = train_data_balanced['cleaned_text']
y_train_balanced = train_data_balanced['label']

# Tokenize data for BERT
print("\nTokenizing data for BERT...")

# Unbalanced data
train_encodings_unbalanced = tokenize_function(X_train_unbalanced.tolist())
# Balanced data
train_encodings_balanced = tokenize_function(X_train_balanced.tolist())

test_encodings = tokenize_function(X_test_ml.tolist())

# Convert labels to tensors
train_labels_unbalanced = torch.tensor(y_train_unbalanced.tolist())
train_labels_balanced = torch.tensor(y_train_balanced.tolist())
test_labels = torch.tensor(y_test_ml.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Function to create data loaders
def create_data_loader(encodings, labels, batch_size=16):
    dataset = TensorDataset(
        encodings['input_ids'], encodings['attention_mask'], labels)
    data_loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
    return data_loader

# Create data loaders
train_dataloader_unbalanced = create_data_loader(train_encodings_unbalanced, train_labels_unbalanced)
train_dataloader_balanced = create_data_loader(train_encodings_balanced, train_labels_balanced)
test_dataset = TensorDataset(
    test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=16)

# Function to train and evaluate BERT model
def train_evaluate_bert(train_dataloader, data_desc):
    # Check if GPU is available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"\nUsing device: {device} for {data_desc} data")

    # Load pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=len(label_encoder.classes_))
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 4
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training function
    def train():
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        avg_loss = total_loss / len(train_dataloader)
        print(f"\nAverage Training Loss: {avg_loss:.4f}")

    # Evaluation function
    def evaluate():
        model.eval()
        predictions, true_labels = [], []
        progress_bar = tqdm(test_dataloader, desc="Evaluating")
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to('cpu').numpy()
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, axis=1).detach().cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels)
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, target_names=label_encoder.classes_, output_dict=True)
        print(f"Accuracy on {data_desc} data: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))
        return accuracy, report

    # Lists to store metrics
    accuracies = []
    reports = []

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} for {data_desc} data")
        train()
        acc, report = evaluate()
        accuracies.append(acc)
        reports.append(report)

    return accuracies, reports

# Train and evaluate BERT on unbalanced data
bert_unbalanced_accuracies, bert_unbalanced_reports = train_evaluate_bert(train_dataloader_unbalanced, "unbalanced")

# Train and evaluate BERT on balanced data
bert_balanced_accuracies, bert_balanced_reports = train_evaluate_bert(train_dataloader_balanced, "balanced")

def add_bert_metrics(metrics_df, bert_reports, bert_accuracies, data_desc):
    # Take the last epoch's metrics
    bert_report = bert_reports[-1]
    # Use pandas.concat instead of append
    metrics_df = pd.concat([metrics_df, pd.DataFrame([{
        'Model': 'BERT',
        'Accuracy': bert_accuracies[-1],
        'Precision': bert_report['weighted avg']['precision'],
        'Recall': bert_report['weighted avg']['recall'],
        'F1-Score': bert_report['weighted avg']['f1-score']
    }])], ignore_index=True)
    return metrics_df


metrics_unbalanced = add_bert_metrics(metrics_unbalanced, bert_unbalanced_reports, bert_unbalanced_accuracies, "unbalanced")
metrics_balanced = add_bert_metrics(metrics_balanced, bert_balanced_reports, bert_balanced_accuracies, "balanced")

# Update plots with BERT metrics
plot_metrics_box(metrics_unbalanced, 'unbalanced')
plot_metrics_bar(metrics_unbalanced, 'unbalanced')

plot_metrics_box(metrics_balanced, 'balanced')
plot_metrics_bar(metrics_balanced, 'balanced')

print("\nAll tasks completed successfully.")