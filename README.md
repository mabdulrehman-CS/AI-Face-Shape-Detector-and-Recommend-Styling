# AI Face Shape Detector and Recommend Styling

**AIFace** is an advanced "Digital Stylist" application. It uses state-of-the-art computer vision to analyze your face shape and provides personalized, professional recommendations for hairstyles, beards, and glasses.

---

## 🌟 Capabilities & Features

### 📸 Input Methods
The system supports two distinct ways to analyze a face:
1.  **Live Camera Feed**: Real-time analysis at 30 FPS. Ideal for quick checks.
2.  **File Upload**: Drag & drop high-resolution images.
    *   **Gender Selector**: You can specify **"Male"** or **"Female"** before analysis. This filters the results (e.g., showing Beards for men, Makeup for women).

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

## ⚙️ Under the Hood: What happens when you click "Analyze"?
Here is the step-by-step journey of your image, explained simply:

1.  **The Snapshot**: You click the camera button. The browser takes a digital photo.
2.  **The Handover**: The photo travels from your browser to our Python Server (the "Kitchen").
3.  **The Geometric Check**: First, **MediaPipe** maps 478 dots on the face. It measures simple things like "Is the face twice as long as it is wide?".
4.  **The AI Intuition**: Simultaneously, the **Deep Learning Model** looks at the photo. It doesn't measure; it *feels*. It recognizes patterns it saw during training (like the curve of a jawline).
5.  **The Council Vote**: The Geometry and the AI compare notes. If they disagree, a "Voting System" decides the winner based on confidence.
6.  **The Expert Advice**: Once the shape is decided (e.g., "Oval"), the system opens its Rulebook (`rules.json`) and picks the correct advice for your gender.
7.  **The Delivery**: The result travels back to your browser and appears as a card.

## 📖 Project File Dictionary 
If you browse the files, here is what each one actually does in plain English:

| File Name | Explanation |
| :--- | :--- |
| `run_app.bat` | Double-click this to start everything. |
| `requirements.txt` | Tells Python which tools (ingredients) to download. |
| `src/app/main.py` | It directs web traffic (your image) to the right place. |
| `src/recommendation/engine.py` | This is where the thinking happens. |
| `src/recommendation/rules.json` | Contains all the grooming knowledge. |
| `models/final_model.keras` | The file where the AI stores what it learned from 7,000 photos. |
| `src/app/static/index.html` | The structure of the webpage you see. |
| `src/app/static/style.css` | Makes the website look dark and modern. |

---

## 🧠 Training Pipeline: How it Learns

The AI was not just "trained once". We used a sophisticated **2-Stage Transfer Learning Strategy** to ensure maximum accuracy without destroying the pre-trained knowledge.

### Stage 1: The "Hasty" Head Training (10 Epochs)
*   **Goal**: Teach the model the *concept* of Face Shapes without confusing it.
*   **Action**: We "Froze" the main brain (EfficientNetV2 backbone). We only trained the final "Head" (the decision layer).
*   **Duration**: **10 Epochs**.
*   **Result**: The model learned coarse differences (e.g., Round vs Square) but missed subtle details.

### Stage 2: The "Fine-Tuning" Surgery (30+ Epochs)
*   **Goal**: Teach the model to see subtle human details (Jawlines, Cheekbones, etc).
*   **Action**: We "Unfroze" the top 30% of the brain. We used a very low learning rate (`1e-5`) to gently nudge the weights.
*   **Duration**: Tuned for **30 to 50 Epochs** with Early Stopping.
*   **Result**: The model achieved high-fidelity accuracy (~74%) on completely unseen test data.

---

## 📂 System Architecture & Files

*   **`src/app/`**: The Web Application.
    *   **Frontend**: Built with **HTML5** and **Vanilla JavaScript**. It handles the webcam and displays the dark-mode UI.
    *   **Backend**: Powered by **FastAPI**. It receives images, runs the AI, and sends back JSON results.
*   **`src/recommendation/`**:
    *   `engine.py`: The Python script that runs the Hybrid Analysis.
    *   `rules.json`: A massive database containing every grooming tip and style rule.
*   **`src/training/`**:
    *   `train.py`: The script used to teach the AI model using Focal Loss and Fine-Tuning.
*   **`models/`**:
    *   Contains `final_model.keras`, the saved brain of the AI.

---

## 🛠️ Requirements & Tech Stack

This project was built using the following robust technologies:

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **AI Core** | **TensorFlow** | The engine behind the Deep Learning model. |
| **Vision** | **MediaPipe** | Used for precise face landmark detection. |
| **Vision** | **OpenCV** | Used for image processing, alignment, and cropping. |
| **Backend** | **FastAPI** | High-performance Python web framework. |
| **Server** | **Uvicorn** | The lightning-fast server launcher. |
| **Data** | **Pandas** | Used for handling dataset CSV attributes. |

---

## 🚀 How to Use

### 1. Installation
First, install the necessary libraries:
```bash
pip install -r requirements.txt
```

### 2. Launching the App
We have made it extremely simple. Just run:
```bash
.\run_app.bat
```
*   This will start the server.
*   It will automatically open your browser to `http://localhost:8001`.

---

**Developed for the User**.
This documentation serves as the complete guide to the project.
