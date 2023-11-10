import cv2
import math
import numpy as np
import urllib.request

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Age estimation model
age_model_url = "https://github.com/GilLevi/AgeGenderEstimation/raw/master/models/age_net.caffemodel"
age_model_path = "age_net.caffemodel"
urllib.request.urlretrieve(age_model_url, age_model_path)

# Gender classification model
gender_model_url = "https://github.com/GilLevi/AgeGenderEstimation/raw/master/models/gender_net.caffemodel"
gender_model_path = "gender_net.caffemodel"
urllib.request.urlretrieve(gender_model_url, gender_model_path)

# Read the video stream frame by frame
while True:
    # Read a single frame from the video stream
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face for age and gender estimation
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Perform age estimation
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = int(math.floor(age_preds[0] * 100))
        
        # Perform gender classification
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = 'Male' if np.argmax(gender_preds[0]) == 0 else 'Female'
        
        # Draw rectangles and text labels on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        label = f'Age: {age}, Gender: {gender}'
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame with detected faces
    cv2.imshow('Face Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()
