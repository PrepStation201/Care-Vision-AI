# Care Vision AI: Brain Tumor Classification 🧠

## Overview

Care Vision AI is a deep learning project that classifies brain tumors from MRI images. It leverages pre-trained models like EfficientNet and provides an interactive web application for real-time predictions. This project demonstrates a complete MLOps pipeline, from data preprocessing and model training to deployment.



[Image of brain MRI scan]


---

## Features

-   **High Accuracy**: Achieves high accuracy in classifying four types of brain MRI images: Glioma Tumor, Meningioma Tumor, No Tumor, and Pituitary Tumor.
-   **Multiple Models**: Experimented with various state-of-the-art models, including EfficientNet and VGG16.
-   **Interactive Web App**: A user-friendly web application built with Streamlit for easy, real-time classification.
-   **Object-Oriented**: The codebase is refactored using OOP principles for better modularity, reusability, and maintainability.
-   **Structured Project**: Follows professional software engineering practices with a `src` layout to separate source code from other project files.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/care-vision-ai.git](https://github.com/your-username/care-vision-ai.git)
    cd care-vision-ai
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Training the Model

-   Place your training and testing data in a directory structure like this:
    ```
    data/
    ├── training/
    │   ├── glioma_tumor/
    │   ├── meningioma_tumor/
    │   ├── no_tumor/
    │   └── pituitary_tumor/
    └── testing/
        ├── glioma_tumor/
        ├── meningioma_tumor/
        ├── no_tumor/
        └── pituitary_tumor/
    ```
-   Update the `TRAIN_DIR` and `TEST_DIR` paths in `src/main.py`.
-   Run the training script:
    ```bash
    python src/main.py
    ```

### 2. Running the Web Application

-   Make sure you have a trained model file (e.g., `EfficientNet_mri_classifier.pth`) in the root directory.
-   Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

---

## Project Structure