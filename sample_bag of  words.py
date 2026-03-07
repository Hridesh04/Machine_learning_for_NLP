## Bag of words implementation for text classification 

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Load spam dataset
def load_spam_dataset():
    """
    Load the SMS Spam Collection dataset
    Download from: https://www.kaggle.com/uciml/sms-spam-collection-dataset
    """
    # If using a local CSV file
    df = pd.read_csv('spam.csv', encoding='latin-1')
    
    # Keep only necessary columns (adjust column names as per your dataset)
    df = df[['v1', 'v2']]  # 'v1' = label (ham/spam), 'v2' = message
    df.columns = ['label', 'message']
    
    # Convert labels to binary (0 = ham, 1 = spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    return df


# Alternative: Create sample data for testing
def create_sample_spam_dataset():
    """Create a small sample dataset for testing"""
    data = {
        'label': [0, 0, 1, 1, 0, 1, 0, 1],
        'message': [
            'Hey, how are you doing today?',
            'Meeting at 5 PM tomorrow',
            'Congratulations! You won a free prize. Click here now!',
            'CLAIM YOUR FREE MONEY NOW! Limited time offer',
            'Can we reschedule the appointment?',
            'You have won 1000 dollars. Claim now!',
            'Great job on the project',
            'Limited time offer! Buy now and get 50% off'
        ]
    }
    return pd.DataFrame(data)


def bag_of_words_classifier(df):
    """
    Implement Bag of Words for spam classification
    """
    # Extract features and labels
    X = df['message']
    y = df['label']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create Bag of Words model
    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words='english',
        lowercase=True,
        min_df=1,
        max_df=0.95
    )
    
    # Transform training and testing data
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    
    # Train Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_bow, y_train)
    
    # Make predictions
    y_pred_train = classifier.predict(X_train_bow)
    y_pred_test = classifier.predict(X_test_bow)
    
    return {
        'vectorizer': vectorizer,
        'classifier': classifier,
        'X_test': X_test,
        'X_test_bow': X_test_bow,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'y_pred_train': y_pred_train,
        'y_train': y_train
    }


def evaluate_model(results):
    """
    Evaluate the model performance
    """
    y_test = results['y_test']
    y_pred_test = results['y_pred_test']
    y_train = results['y_train']
    y_pred_train = results['y_pred_train']
    
    print("=" * 60)
    print("BAG OF WORDS - SPAM CLASSIFICATION RESULTS")
    print("=" * 60)
    
    print("\n--- TRAINING SET METRICS ---")
    print(f"Training Accuracy:  {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Training Precision: {precision_score(y_train, y_pred_train):.4f}")
    print(f"Training Recall:    {recall_score(y_train, y_pred_train):.4f}")
    
    print("\n--- TESTING SET METRICS ---")
    print(f"Testing Accuracy:   {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"Testing Precision:  {precision_score(y_test, y_pred_test):.4f}")
    print(f"Testing Recall:     {recall_score(y_test, y_pred_test):.4f}")
    
    print("\n--- CONFUSION MATRIX ---")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred_test, target_names=['Ham', 'Spam']))
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix - Spam Classification')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def predict_message(message, results):
    """
    Predict if a single message is spam or ham
    """
    vectorizer = results['vectorizer']
    classifier = results['classifier']
    
    message_bow = vectorizer.transform([message])
    prediction = classifier.predict(message_bow)[0]
    probability = classifier.predict_proba(message_bow)[0]
    
    label = "SPAM" if prediction == 1 else "HAM"
    spam_confidence = probability[1] * 100
    
    print(f"\nMessage: {message}")
    print(f"Prediction: {label}")
    print(f"Spam Confidence: {spam_confidence:.2f}%")
    
    return prediction


# Main execution
if __name__ == "__main__":
    # Load dataset (using sample data for testing)
    print("Loading spam dataset...")
    df = create_sample_spam_dataset()
    # Uncomment below to use actual dataset
    # df = load_spam_dataset()
    
    print(f"Dataset size: {len(df)} messages")
    print(f"Spam count: {df['label'].sum()}")
    print(f"Ham count: {len(df) - df['label'].sum()}\n")
    
    # Train Bag of Words classifier
    print("Training Bag of Words classifier...")
    results = bag_of_words_classifier(df)
    
    # Evaluate model
    evaluate_model(results)
    
    # Test with sample messages
    print("\n" + "=" * 60)
    print("TESTING WITH SAMPLE MESSAGES")
    print("=" * 60)
    test_messages = [
        "Can you call me tomorrow?",
        "Claim your free iPhone now! Limited offer!",
        "Let's meet for lunch",
        "WINNER! You've been selected for a prize!"
    ]
    
    for msg in test_messages:
        predict_message(msg, results)
