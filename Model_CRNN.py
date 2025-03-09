import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


def load_data(features_path, labels_path):
    """
    Load feature and label datasets.
    """
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    return features.values, labels.values.ravel()


def preprocess_data(features, labels):
    """
    Encode labels and split into train and test sets.
    """
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)  # One-hot encoding

    # Split into train (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_encoder


def build_crnn_model(input_shape, num_classes):
    """
    Build the CRNN model.
    """
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_confusion_matrix(cm, labels, title, save_path):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_pred, save_path):
    """
    Plot and save ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true[:, 1], y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.close()


def save_classification_report(y_true, y_pred, label_encoder, save_path):
    """
    Save classification report.
    """
    report = classification_report(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), target_names=label_encoder.classes_)
    with open(save_path, 'w') as f:
        f.write(report)


def main():
    # Paths
    features_path = r"E:\Personal\Fall_Detection\archive\Features\features.csv"
    labels_path = r"E:\Personal\Fall_Detection\archive\Features\labels.csv"
    results_dir = r"E:\Personal\Fall_Detection\archive\Results"
    os.makedirs(results_dir, exist_ok=True)

    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    features, labels = load_data(features_path, labels_path)
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(features, labels)

    # Reshape data for Conv1D
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Build and train CRNN model
    print("🛠️ Building and training CRNN model...")
    model = build_crnn_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=y_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping])

    # Save model
    model_save_path = os.path.join(results_dir, "model.h5")
    model.save(model_save_path)
    print(f"✅ Model saved to {model_save_path}")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Confusion Matrix
    plot_confusion_matrix(confusion_matrix(np.argmax(y_train, axis=1), np.argmax(y_train_pred, axis=1)),
                          label_encoder.classes_, 'Train Confusion Matrix',
                          os.path.join(results_dir, 'train_confusion_matrix.png'))

    plot_confusion_matrix(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_test_pred, axis=1)),
                          label_encoder.classes_, 'Test Confusion Matrix',
                          os.path.join(results_dir, 'test_confusion_matrix.png'))

    # ROC Curves
    plot_roc_curve(y_train, y_train_pred, os.path.join(results_dir, 'train_roc_curve.png'))
    plot_roc_curve(y_test, y_test_pred, os.path.join(results_dir, 'test_roc_curve.png'))

    # Classification Report
    save_classification_report(y_test, y_test_pred, label_encoder, os.path.join(results_dir, 'classification_report.txt'))

    print(f"✅ All results saved to {results_dir}")


if __name__ == "__main__":
    main()
