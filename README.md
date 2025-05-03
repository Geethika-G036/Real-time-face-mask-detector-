# Real-time-face-mask-detector-

The objective of this AI project is to develop a real-time face mask detection system using Python, computer vision, and deep learning techniques . The system aims to identify individuals wearing or not wearing face masks through live video feed, provide visual feedback with bounding boxes and labels, and trigger audio alerts for non-compliance. This solution is designed to promote safety and adherence to mask-wearing protocols in public or workplace environments.

This project implements a real-time face mask detection system using Python and popular libraries such as OpenCV, TensorFlow, Keras and Pygame. The system leverages a pre-trained deep learning model to classify whether a person is wearing a mask or not. It uses a webcam feed to detect faces, processes the detected face images, and predicts the mask status in real-time.
Key features of this project include:
# Face Detection: The Haar-Cascade Classifier is used to detect faces in the video feed.
# Mask Classification: A pre-trained Keras model (`model2.h5`) classifies the detected face as "Mask" or "No Mask.“
# Real-Time Alerts: If a person is detected without a mask, a red rectangle is drawn around their face, and an audio alert is played using Pygame.
# Timestamp Overlay: The current date and time are displayed on the video feed for additional context.
This project demonstrates the integration of computer vision, machine learning, and multimedia libraries to create a practical and interactive application. It can be used in public spaces or workplaces to promote safety and compliance with mask-wearing guidelines.
