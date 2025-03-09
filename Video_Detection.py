import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder
from twilio.rest import Client

# Twilio Configuration
TWILIO_ACCOUNT_SID = 'ACcb4c212a5a082305f8465e464d86e928'
TWILIO_AUTH_TOKEN = '65375828cbee5dd0ff012c76dcd9aa59'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'  # Twilio WhatsApp Sandbox number
USER_WHATSAPP_NUMBER = 'whatsapp:+916369432630'  # Replace with your verified WhatsApp number


def send_whatsapp_alert():
    """
    Send an alert message via WhatsApp when a fall is detected.
    """
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body='⚠️ Alert: Fall detected! Immediate attention required.',
            from_=TWILIO_WHATSAPP_NUMBER,
            to=USER_WHATSAPP_NUMBER
        )
        print(f"📲 WhatsApp alert sent successfully. SID: {message.sid}")
    except Exception as e:
        print(f"❌ Failed to send WhatsApp alert: {e}")


def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocess a video frame for feature extraction.
    """
    frame = cv2.resize(frame, target_size)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return frame


def extract_features(frame, feature_extractor):
    """
    Extract features from a video frame using EfficientNet.
    """
    features = feature_extractor.predict(frame)
    return features.flatten()


def classify_frame(model, features, label_encoder):
    """
    Classify a video frame using the trained CRNN model.
    """
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    features = np.expand_dims(features, axis=-1)  # Add channel dimension for Conv1D compatibility

    prediction = model.predict(features)
    predicted_label = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(predicted_label)[0]


def process_video(video_path, model, feature_extractor, label_encoder):
    """
    Process a video, classify frames, and detect falls.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open the video file.")
        return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of the video

        frame_count += 1

        # Preprocess and extract features from the current frame
        processed_frame = preprocess_frame(frame)
        features = extract_features(processed_frame, feature_extractor)

        # Classify the frame
        result = classify_frame(model, features, label_encoder)

        # Display result on the video frame
        label_text = f"Frame {frame_count}: {result}"
        if result == "Fall":
            label_text += " ⚠️ PERSON HAS FALLEN!"

            # Display on frame
            cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Fall Detection', frame)

            # Send WhatsApp Alert and terminate the program
            print("\n⚠️ ALERT: Fall detected. Sending WhatsApp alert...")
            send_whatsapp_alert()
            print("\n🛑 Fall detected. Terminating the program.")
            break  # Exit the loop immediately after detecting a fall

        cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Fall Detection', frame)

        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    # Paths
    model_path = r"E:\Personal\Fall_Detection\Results\model.h5"
    label_encoder_path = r"E:\Personal\Fall_Detection\Features\labels.csv"
    video_path = input("Enter the path of the video file to classify: ").strip()

    if not os.path.isfile(video_path):
        print("❌ Invalid video path. Please check and try again.")
        return

    # Load Model
    print("📊 Loading the trained model...")
    model = load_model(model_path)

    # Load Label Encoder
    print("📊 Loading label encoder...")
    labels = pd.read_csv(label_encoder_path).values.ravel()
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # Feature Extractor (EfficientNet)
    print("📊 Loading feature extractor (EfficientNet)...")
    feature_extractor = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), pooling='avg')

    # Process the video
    print("🛠️ Processing the video for fall detection...")
    process_video(video_path, model, feature_extractor, label_encoder)


if __name__ == "__main__":
    main()
