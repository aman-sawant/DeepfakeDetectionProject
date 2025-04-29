Deepfake Detection System

A cross-platform application for detecting deepfake audio and image files using machine learning. The system combines a JavaFX desktop client with a Flask-based Python REST API backend for real-time deepfake prediction.

Features

-Detects fake or real audio files (WAV) using MFCC feature extraction and a deep learning model.

-Detects fake or real image files (JPEG/PNG) using a pretrained CNN model.

-JavaFX GUI built with OkHttp and Gson for smooth file upload and response parsing.

-Python Flask server serves predictions using TensorFlow models.

-Real-time confidence score for prediction interpretation.


Technologies Used

•Backend (Python + Flask)
-TensorFlow / Keras
-Librosa (for audio feature extraction)
-OpenCV / PIL (for image preprocessing)
-Flask REST API


•Frontend (Java)
-JavaFX for GUI
-TarsosDSP (for audio handling, if needed)
-OkHttp (for HTTP requests)
-Gson (for JSON parsing)

How to Run

1. Start the Flask Server

cd "Model training and flask"
pip install -r requirements.txt
python server.py

Server runs at http://127.0.0.1:5000/

2. Run JavaFX Client

Import the project in IntelliJ or any Java IDE. Ensure you have Java 11+ and Gradle support.

cd "Java Client"
./gradlew run

3. Use the Application

Click "Upload Audio" to test a WAV file.

Click "Upload Image" to test a JPEG/PNG file.

The system shows prediction with confidence.


Screenshots

(Add relevant UI screenshots here)

Limitations

-.h5 model files are not included due to GitHub's 100MB file limit.

-You may need to manually download them or retrain the models.


License

-This project is for educational and research purposes. No commercial use permitted without permission.

