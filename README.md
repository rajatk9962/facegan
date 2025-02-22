# **Facial Image Generation using GANs**

## Introduction

This project uses **Generative Adversarial Networks (GANs)** to generate **photorealistic human face images**. The model allows for **custom feature adjustments**, enabling users to control attributes such as **smiling, hair color, facial hair, and more**. A **Flask web app** is included for easy interaction.

---

## Features

- ✅ **Generate random human faces**
- ✅ **Modify facial features dynamically**
- ✅ **Pre-trained GAN model for high-quality outputs**
- ✅ **Flask-based web interface**
- ✅ **Supports interactive feature manipulation**

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/facial-gan.git
   cd facial-gan
   ```
2. **Install dependencies**:
   ```bash
    pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
    python app.py
   ```
4. Access the web app at:
   ```bash
   http://127.0.0.1:5000/
   ```

## Usage
- Click "Generate random image" to create a new face.
- Select a feature (e.g., smiling, bald, mustache) and adjust its intensity.
- View and save generated images.



## File Structure

📁 facial-gan
├── app.py               # Flask web application
├── main.py              # Core logic for face generation
├── Layers.py            # Neural network layers (Generator & Classifier)
├── Subfunctions.py      # Helper functions for noise and scoring
├── generator50.pth      # Pre-trained GAN model
├── classifierv6.pth     # Pre-trained classifier
├── templates/
│   ├── index.html       # Web UI for the application
├── static/              # Stores generated images
├── requirements.txt     # Required Python packages
└── README.md            # Project documentation
