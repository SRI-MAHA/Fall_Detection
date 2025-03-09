import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model


def load_model():
    """
    Load the EfficientNet model pre-trained on ImageNet without the classification head.
    """
    base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model


def extract_features(image_path, model, image_size=(224, 224)):
    """
    Extract features from a single image using EfficientNet.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read {image_path}. Skipping.")
        return None

    image = cv2.resize(image, image_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    features = model.predict(image)
    return features.flatten()


def process_directory(input_dir, label, model, feature_list, label_list):
    """
    Process all images in a directory, extract features, and store them.
    """
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        if os.path.isfile(image_path):
            features = extract_features(image_path, model)
            if features is not None:
                feature_list.append(features)
                label_list.append(label)


def main():
    # Directories
    fall_dir = r"E:\Personal\Fall_Detection\Preprocessed\Fall"
    not_fall_dir = r"E:\Personal\Fall_Detection\Preprocessed\Not_Fall"
    output_dir = r"E:\Personal\Fall_Detection\Features"

    os.makedirs(output_dir, exist_ok=True)

    # Load the EfficientNet model
    print("📥 Loading EfficientNet model...")
    model = load_model()
    print("✅ Model loaded successfully!")

    # Lists to store features and labels
    features = []
    labels = []

    # Process Fall images
    print("\n📊 Processing 'Fall' images...")
    process_directory(fall_dir, label='Fall', model=model, feature_list=features, label_list=labels)

    # Process Not_Fall images
    print("\n📊 Processing 'Not_Fall' images...")
    process_directory(not_fall_dir, label='Not_Fall', model=model, feature_list=features, label_list=labels)

    # Save features and labels to CSV
    print("\n💾 Saving features and labels to CSV files...")
    features_df = pd.DataFrame(features)
    labels_df = pd.DataFrame(labels, columns=['Label'])

    features_path = os.path.join(output_dir, 'features.csv')
    labels_path = os.path.join(output_dir, 'labels.csv')

    features_df.to_csv(features_path, index=False)
    labels_df.to_csv(labels_path, index=False)

    print("✅ Features and labels saved successfully!")
    print(f"📂 Features file: {features_path}")
    print(f"📂 Labels file: {labels_path}")


if __name__ == "__main__":
    main()
