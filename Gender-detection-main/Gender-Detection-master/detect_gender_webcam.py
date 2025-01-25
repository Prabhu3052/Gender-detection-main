from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# Load the pre-trained gender detection model
model = load_model("C:/Users/pp684/Videos/GitHub/Gender-detection-main/Gender-detection-main/Gender-Detection-master/Model folder/gender_detection.h5")

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Define the class labels
classes = ['man', 'woman']

# Loop to process webcam frames
while webcam.isOpened():
    # Read frame from webcam
    status, frame = webcam.read()

    if not status:
        print("Could not read frame. Exiting...")
        break

    # Detect faces in the frame
    faces, confidences = cv.detect_face(frame)

    # Process each detected face
    for idx, face in enumerate(faces):
        (startX, startY, endX, endY) = face

        # Ensure face boundaries are within the frame
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(frame.shape[1], endX), min(frame.shape[0], endY)

        # Draw rectangle around the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop and preprocess the face
        face_crop = frame[startY:endY, startX:endX]
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocessing: Resize, normalize, and convert channel order
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if needed
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Predict gender
        prediction = model.predict(face_crop)[0]  # Get the prediction for the face
        man_confidence, woman_confidence = prediction[0], prediction[1]
        idx = np.argmax(prediction)  # Get the index of the class with highest confidence
        label = classes[idx]

        # Format the label
        label_text = f"{label}: {prediction[idx] * 100:.2f}%"

        # Debug: Print confidence scores
        print(f"Man confidence: {man_confidence * 100:.2f}%, Woman confidence: {woman_confidence * 100:.2f}%")

        # Position the label above the face rectangle
        label_y = startY - 10 if startY - 10 > 10 else startY + 10

        # Add label to the frame
        cv2.putText(frame, label_text, (startX, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Gender Detection", frame)

    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
