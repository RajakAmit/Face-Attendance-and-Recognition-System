Face Attendance and Recognition System
This project implements an automated Face Recognition and Attendance System using deep learning techniques for facial identification. The system utilizes OpenCV, FaceNet embeddings, and machine learning classifiers like K-Nearest Neighbors (KNN) and Support Vector Machine (SVM). The goal is to enable real-time face recognition and attendance logging based on facial embeddings.

Key Features
Real-time Face Detection: Detects and extracts faces from video frames using OpenCV's DNN (Deep Neural Networks).
Face Embedding Extraction: Uses FaceNet to extract 128-dimensional embeddings that uniquely represent an individual's facial features.
Person Identification: Implements KNN and SVM classifiers to recognize individuals based on facial embeddings.
Attendance Logging: Automatically records the presence of individuals by identifying their faces in real-time.
Model Evaluation: Provides performance metrics (precision, recall, F1-score, and accuracy) for KNN and SVM classifiers.
Model Persistence: Saves trained models (KNN, SVM) using joblib for reusability.
Visualization: Displays images with predicted labels and confidence scores using matplotlib.
Technologies Used
Programming Languages: Python
Libraries:
OpenCV: For real-time face detection and video capture.
Keras (FaceNet): For face embedding extraction.
NumPy: For data manipulation and preprocessing.
scikit-learn: For KNN, SVM, and model evaluation.
Matplotlib: For visualizing images and results.
joblib: For saving and loading trained models.
Setup & Installation
To run the project locally, you need to install the following dependencies:

pip install
opencv-python
keras
numpy 
scikit-learn
matplotlib
joblib
Additionally, need the FaceNet model to extract facial embeddings. You can use the pre-trained model available through the keras_facenet library.

Usage
1. Video Capture & Dataset Creation
The system captures real-time video from a camera and processes the frames to extract faces.
You can use the script to store captured faces as part of a custom dataset, which is then used for training the model.
2. Face Detection
Face detection is performed using OpenCV's pre-trained DNN model. The system detects faces in the video stream and extracts bounding boxes around the faces.
3. Face Embedding Extraction
The detected faces are passed through FaceNet, a pre-trained deep learning model, to generate unique 128-dimensional embeddings that represent each person's facial features.
4. Model Training
The face embeddings are used to train the following classifiers:
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Both classifiers are trained on the embeddings, with 80% of the dataset used for training and 20% for testing.
5. Model Evaluation
After training the models, the performance is evaluated using classification metrics such as accuracy, precision, recall, and F1-score.
6. Real-Time Recognition
The system detects faces in real-time from a live video feed, extracts embeddings, and classifies the faces using the trained KNN and SVM models.
The system logs attendance based on the predictions and outputs a probability score for each prediction.
7. Model Persistence
The trained models are saved using joblib to avoid retraining each time the system is run. You can load the pre-trained models from disk to make predictions.
