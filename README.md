# AI Face Shape Detector and Recommend Styling

**AIFace** is an advanced "Digital Stylist" application. It uses state-of-the-art computer vision to analyze your face shape and provides personalized, professional recommendations for hairstyles, beards, and glasses.

---

## 🌟 Capabilities & Features

### 📸 Input Methods
The system supports two distinct ways to analyze a face:
1.  **Live Camera Feed**: Real-time analysis at 30 FPS. Ideal for quick checks.
2.  **File Upload**: Drag & drop high-resolution images.
    *   **Gender Selector**: You can specify **"Male"** or **"Female"** before analysis. This filters the results (e.g., showing Beards for men, Makeup for women).

### 🔒 Smart Gender Validation
To ensure accurate and relevant recommendations, the system includes **AI-powered gender validation**:
*   **DeepFace Integration**: Uses the DeepFace library for accurate gender detection from facial features.
*   **Automatic Mismatch Detection**: If you select "Male" but upload a female photo (or vice versa), the system will reject the image with a helpful error message.
*   **Why This Matters**: Prevents inappropriate recommendations (e.g., beard styles for women) and ensures you get styling advice relevant to your gender.

---

### 🔍 Analysis Outputs
The system doesn't just guess; it provides mathematical proof:
*   **Predicted Shape**: (e.g., "Heart", "Square", "Oval", "Round", "Oblong").
*   **Confidence Score**: A percentage (e.g., "98.5% Confident") based on the neural network's certainty.
*   **Visual Recommendations**: Cards showing specific styles that suit your face for Males and Females.
*   **Performance Metrics** (For Developers):
    *   **Confusion Matrix**: A heatmap showing exactly where the model makes mistakes (e.g., "Confused Oblong for Oval").
    *   **Detailed Report**: Calculates **Precision**, **Recall**, and **F1-Score** for every single class.


---

## 🌎 Real World Value (Why does this matter?)
Most people struggle to find a haircut or beard style that suits them. They often pick what looks good on a celebrity, only to be disappointed. 
*   **The Problem**: "One size fits all" doesn't work for faces.
*   **The Solution**: This AI acts as an objective, mathematical consultant. It tells you *why* a style works (e.g., "This beard hides your sharp jaw" or "This haircut balances your round face").

---

## ⚙️ What happens when you click "Analyze"?
Here is the step-by-step journey of your image, explained simply:

1.  **The Snapshot**: You click the camera button. The browser takes a digital photo.
2.  **The Handover**: The photo travels from your browser to our Python Server.
3.  **The Geometric Check**: First, **MediaPipe** maps 478 dots on the face. It measures simple things like "Is the face twice as long as it is wide?".
4.  **The AI Intuition**: Simultaneously, the **Deep Learning Model** looks at the photo. It doesn't measure; it *feels*. It recognizes patterns it saw during training (like the curve of a jawline).
5.  **The Council Vote**: The Geometry and the AI compare notes. If they disagree, a "Voting System" decides the winner based on confidence.
6.  **The Expert Advice**: Once the shape is decided (e.g., "Oval"), the system opens its Rulebook (`rules.json`) and picks the correct advice for your gender.
7.  **The Delivery**: The result travels back to your browser and appears as a card.

---

## 🧠 Training Pipeline: How it Learns

The AI was not just "trained once". We used a sophisticated **2-Stage Transfer Learning Strategy** to ensure maximum accuracy without destroying the pre-trained knowledge.

### Stage 1: The Head Training (10 Epochs)
*   **Goal**: Teach the model the *concept* of Face Shapes without confusing it.
*   **Action**: We "Froze" the main brain (EfficientNetV2 backbone). We only trained the final "Head" (the decision layer).
*   **Duration**: **10 Epochs**.
*   **Result**: The model learned coarse differences (e.g., Round vs Square) but missed subtle details.

### Stage 2: The "Fine-Tuning" Surgery (40 Epochs)
*   **Goal**: Teach the model to see subtle human details (Jawlines, Cheekbones, etc).
*   **Action**: We "Unfroze" the top 30% of the brain. We used a very low learning rate (`1e-5`) to gently nudge the weights.
*   **Duration**: Tuned for **40 Epochs** with Early Stopping and model checkpointing.
*   **Result**: The model achieved **83.4% validation accuracy** on completely unseen test data.

---

### Model Versions Available
| Model | Epochs | Validation Accuracy | Description |
| :--- | :---: | :---: | :--- |
| **Fine-Tuned v3 (Best)** | 40 | **83.4%** | Latest model with improved Oval/Round classification |
| Fine-Tuned v2 | 30 | 74.1% | Previous best model |
| Fine-Tuned v1 | 20 | ~70% | Initial fine-tuned model |
| Head Model | 10 | ~60% | Stage 1 baseline |

---

## 📂 System Architecture & Files

```
AIFace/
├── src/
│   ├── app/                    # Web Application
│   │   ├── main.py             # FastAPI backend server
│   │   └── static/             # Frontend assets
│   │       ├── index.html      # Main webpage
│   │       ├── script.js       # JavaScript logic
│   │       └── style.css       # Dark mode styling
│   ├── recommendation/
│   │   ├── engine.py           # Hybrid Analysis engine
│   │   └── rules.json          # Grooming knowledge database
│   ├── training/
│   │   └── train.py            # Model training script
│   └── features/
│       └── landmarks.py        # Face landmark utilities
├── models/
│   ├── final_model.keras       # Default model
│   ├── checkpoints/            # Model versions
│   │   ├── best_fine_v3.keras  # Latest (83.4% accuracy)
│   │   ├── best_fine_v2.keras  # Previous version
│   │   └── best_fine.keras     # v1 model
│   └── metrics/                # Performance metrics JSON
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned images
│   └── splits/                 # Train/Val/Test splits
├── requirements.txt            # Python dependencies
└── run_app.bat                 # Quick launcher
```

*   **`src/app/`**: The Web Application.
    *   **Frontend**: Built with **HTML5** and **Vanilla JavaScript**. It handles the webcam and displays the dark-mode UI.
    *   **Backend**: Powered by **FastAPI**. It receives images, runs the AI, and sends back JSON results.
*   **`src/recommendation/`**:
    *   `engine.py`: The Python script that runs the Hybrid Analysis.
    *   `rules.json`: A massive database containing every grooming tip and style rule.
*   **`src/training/`**:
    *   `train.py`: The script used to teach the AI model using Focal Loss and Fine-Tuning.
*   **`models/`**:
    *   Contains multiple model versions with `best_fine_v3.keras` being the current best performer.

---

## 🛠️ Requirements & Tech Stack

This project was built using the following robust technologies:

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **AI Core** | **TensorFlow** | The engine behind the Deep Learning model. |
| **Gender Detection** | **DeepFace** | Accurate gender validation from facial features. |
| **Vision** | **MediaPipe** | Used for precise face landmark detection. |
| **Vision** | **OpenCV** | Used for image processing, alignment, and cropping. |
| **Backend** | **FastAPI** | High-performance Python web framework. |
| **Server** | **Uvicorn** | The lightning-fast server launcher. |
| **Data** | **Pandas** | Used for handling dataset CSV attributes. |

---

## 🚀 How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/mabdulrehman-CS/AI-Face-Shape-Detector-and-Recommend-Styling.git
cd AI-Face-Shape-Detector-and-Recommend-Styling
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# or
source .venv/bin/activate      # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
We have made it extremely simple. Just run:
```bash
.\run_app.bat
```
Or manually:
```bash
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8001
```
*   This will start the server.
*   Open your browser to `http://localhost:8001`.

---

## 📊 Model Performance

The current best model (**Fine-Tuned v3**) achieves:

| Metric | Score |
| :--- | :---: |
| **Overall Accuracy** | 83.4% |
| **Macro F1 Score** | 0.83 |

### Per-Class Performance
| Face Shape | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Heart | 0.89 | 0.78 | 0.83 |
| Oblong | 0.87 | 0.82 | 0.84 |
| Oval | 0.76 | 0.73 | 0.74 |
| Round | 0.72 | 0.79 | 0.75 |
| Square | 0.91 | 0.89 | 0.90 |

---

## 🔄 Continue Training

To fine-tune the model further:
```bash
python -m src.training.train --epochs 10 --resume
```

---

## 📸 Screenshots

The application provides:
- Real-time webcam face shape detection
- File upload analysis
- Confidence scores with visual feedback
- Personalized grooming recommendations
- Model performance metrics dashboard with confusion matrix

---

**Documentation**.

This documentation serves as the complete guide to the project.

---