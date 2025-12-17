# Bird-Sound-Classification-using-CNN

This project implements an end-to-end Bird Sound Classification system using Deep Learning techniques.
The system classifies bird species based on their audio vocalizations by extracting MFCC (Mel-Frequency Cepstral Coefficients) features and training a Convolutional Neural Network (CNN).

The trained model is deployed using **Streamlit**, allowing users to upload an audio file (`.mp3` or `.wav`) and instantly identify the corresponding bird species along with a reference image.

---

## 🎯 Objectives

* Convert bird audio signals into numerical representations using MFCCs
* Train a CNN model to classify **114 bird species**
* Achieve robust performance on unseen test data
* Deploy the trained model using a **user-friendly web interface**
* Enable real-time bird species prediction from audio input

---

## 📂 Dataset

* **Source:** Kaggle
* **Dataset Name:** *Sound of 114 Species of Birds till 2022*
* **Format:** `.mp3` audio files
* **Structure:**

  ```
  Voice of Birds/
  ├── Andean Guan_sound/
  ├── Asian Koel_sound/
  ├── ...
  └── 114 bird species folders
  ```

Each folder contains multiple bird call recordings belonging to a single species.

---

## 🧹 Data Preprocessing

* Audio loading using `librosa`
* Feature extraction using **MFCCs (40 coefficients)**
* Mean aggregation across time
* Conversion to TensorFlow tensors
* Label encoding of bird species
* Creation of a **prediction dictionary (JSON)** for inference

---

## 🧠 Model Architecture

A **1D Convolutional Neural Network (CNN)** is used to process MFCC features.

### Model Highlights:

* Input shape: `(40, 1)`
* 3 Conv1D layers with Batch Normalization
* MaxPooling for feature reduction
* Fully connected Dense layers with **L2 regularization**
* Dropout for overfitting control
* Softmax output layer for **114 classes**

---

## ⚙️ Training Configuration

* **Optimizer:** Adam
* **Learning Rate:** `1e-5`
* **Loss Function:** Sparse Categorical Crossentropy
* **Batch Size:** 32
* **Epochs:** 250
* **Dataset Split:**

  * Training: 80%
  * Validation: 10%
  * Testing: 10%

---

## 📊 Model Evaluation

* Accuracy and loss monitored during training
* Validation used to prevent overfitting
* Final evaluation performed on test dataset
* Training vs validation accuracy and loss visualized using Matplotlib

---

## 🚀 Deployment (Streamlit App)

The trained model is deployed using **Streamlit** for real-time predictions.

### App Features:

* Upload `.mp3` or `.wav` audio files
* Automatic MFCC extraction
* Bird species prediction
* Display of corresponding bird image
* Clean and centered UI layout

---

## 📁 Generated Files

| File                | Description                            |
| ------------------- | -------------------------------------- |
| `model.h5`          | Trained CNN model                      |
| `prediction.json`   | Mapping of class indices to bird names |
| `Inference_Images/` | Bird images for prediction display     |
| `app.py`            | Streamlit application code             |

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* Librosa
* NumPy, Pandas
* Matplotlib
* OpenCV
* Streamlit
* Google Colab
* Kaggle API

---

## ▶️ How to Run the Project

### Training (Google Colab)

1. Upload `kaggle.json`
2. Download the dataset
3. Run all training cells
4. Save `model.h5` and `prediction.json`

### Deployment (Local / Cloud)

```bash
pip install streamlit librosa tensorflow numpy opencv-python
streamlit run app.py
```

---

## 📌 Applications

* Wildlife monitoring
* Bird species identification
* Biodiversity conservation
* Bioacoustics research
* Eco-acoustic studies

---

## 📜 Conclusion

This project demonstrates how **deep learning and audio signal processing** can be effectively combined to classify bird species from sound.
The deployed system provides an intuitive interface for real-world usage and can be extended to include more species or real-time audio recording.
