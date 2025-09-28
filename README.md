# Sign Language Recognition

A real-time **Sign Language Recognition system** using Python, MediaPipe, and Machine Learning.  
This project recognizes American Sign Language (ASL) hand gestures from a webcam feed.

---

## Features

- Real-time hand landmark detection using **MediaPipe Hands**.
- Recognizes ASL letters with a trained **RandomForest classifier**.
- Simple pipeline for **data collection, model training, and real-time prediction**.
- Easily extendable to neural network models for better accuracy or sentence recognition.

---

## Folder Structure

```

sign_language_recognition/
│
├─ data/                      # Hand landmark .npy files
├─ collect_data.py            # Script for collecting hand landmarks
├─ train_model.py             # Script for training the classifier
├─ real_time_prediction.py    # Script for real-time prediction
├─ sign_language_model.pkl    # Saved trained model
└─ requirements.txt           # Project dependencies

````

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/sign_language_recognition.git
cd sign_language_recognition
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Collect Data

Run the script to capture hand landmarks:

```bash
python collect_data.py
```

* Press `q` to quit the webcam feed.
* Update the `label` variable in the script to collect data for different letters.

---

### 2. Train Model

Train the RandomForest classifier with collected data:

```bash
python train_model.py
```

* The trained model will be saved as `sign_language_model.pkl`.

---

### 3. Real-Time Prediction

Run the real-time recognition script:

```bash
python real_time_prediction.py
```

* The webcam will show the predicted ASL letter on the screen.
* Press `q` to quit.

---

## Dependencies

* Python 3.10+
* OpenCV
* MediaPipe
* NumPy
* scikit-learn
* joblib
* TensorFlow (optional for neural network upgrades)

---

## Future Improvements

* Use **CNN + LSTM** for recognizing sequences of gestures (full words/sentences).
* Add support for more letters and dynamic gestures.
* Integrate with speech synthesis to "speak" recognized words.

---

