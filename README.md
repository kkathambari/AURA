# AURA: Automated Behavioral Screening & Temporal Analysis System

AURA is an advanced, computer-vision-based screening tool designed to detect and quantify Motor Stereotyped Behaviors (MSBs) associated with Autism Spectrum Disorder (ASD). By replacing subjective timing and manual observation with objective, deep-learning-based temporal action recognition, AURA acts as an accessible screening aid.

The system is designed with a **Privacy-First (Edge AI)** architecture, performing all frame processing, spatial feature extraction, and temporal classifications entirely on the user's local machine, requiring no internet connection.

---

## 🚀 Key Features

- **Spatial-Temporal Dual Neural Net**: Processes raw video by extracting spatial representations via ResNet-18, then analyzes sequence frequencies using a Temporal Convolutional Network (TCN).
- **Multi-Stage False Positive Prevention**: Includes real-time screening filters that validate human presence, check motion variance, reject digital screen captures, and perform FFT-based periodicity tests to filter noise.
- **Smart Voting Decision Strategy**: Accumulates temporal class predictions over sliding temporal windows to detect mixed or transient stimming patterns with high statistical accuracy.
- **Dual UI Dashboards**: Features a standard clinical desktop analysis panel (`desktop_app.py`) integrated with a modern, mobile-styled launcher interface (`aura_app.py`).
- **Automated Report Exporter**: Automatically generates structured, print-ready PDF reports and dark-themed CSS-styled HTML summary dashboards.

---

## 🛠️ System Architecture & Workflow

### 1. System Pipeline Architecture
```text
+------------------------------------------------------------+
|                       USER INTERFACE                       |
|  - desktop_app.py (Standard Tkinter Desktop Dashboard)    |
|  - aura_app.py (CustomTkinter Mobile-styled Launcher)     |
+------------------------------------------------------------+
       |                                              ^
(Loads Video)                                   (Displays Report /
       |                                          Video Canvas)
       v                                              |
+------------------------------------------------------------+
|                       WORKER THREAD                        |
|  - Multi-threaded execution to prevent GUI freezing        |
|  - Thread-safe Progress Queue communication                |
+------------------------------------------------------------+
       |                                              ^
(Processes Frames)                              (Inference Results)
       v                                              |
+------------------------------------------------------------+
|                      INFERENCE ENGINE                      |
|  - Spatial Feature Extractor (ResNet-18)                   |
|  - Temporal Convolutional Network (TCN)                    |
|  - Smart Voting & Proportional Consensus Logic             |
+------------------------------------------------------------+
       |                                              ^
(Spatial Features)                              (Rhythm/Periodicity)
       v                                              |
+------------------------------------------------------------+
|             FALSE POSITIVE PREVENTION LAYER                |
|  - Screen Recording Check (Pixel Variance/Morphology)      |
|  - Human Detection (Haar Cascades, HOG Person Detector)    |
|  - Motion Energy (Mean & Std Dev Frame-Differencing)       |
|  - Rhythmicity analysis (FFT Power Spectral Density)      |
+------------------------------------------------------------+
       |
(Finalized Screening Statistics)
       v
+------------------------------------------------------------+
|                      REPORT GENERATOR                      |
|  - report_generator.py (PDF Engine via fpdf2)              |
|  - HTML Summary Exporter (Web-viewable Report)             |
+------------------------------------------------------------+
```

### 2. Inference Data Flow
```text
[Raw Input Video] -> [Preprocessing (6 FPS)] -> [FP Rejection Filters] ->
  [ResNet-18 Backbone] -> [Temporal Window Buffer (10 Frames)] ->
    [TCN Classifier] -> [FFT Rhythm Verification] -> [Smart Voting] ->
      [Screening Verdict Mapping] -> [PDF & HTML Reports]
```

---

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kkathambari/AURA.git
   cd AURA
   ```

2. **Set up virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Application

### 1. Launch the Main AURA Application
Run the modern launcher with service shortcuts, splash screens, and target tracking:
```bash
python aura_app.py
```

### 2. Run the Expert Screening Interface
Run the standard analysis dashboard directly:
```bash
python desktop_app.py
```

### 3. Quantitative Model Evaluation
Run metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix) on the unseen test dataset:
```bash
python evaluate_model.py
```

---

## 📊 Evaluation Metrics
Evaluating the trained TCN model on the unseen validation splits yields:
- **Overall Accuracy**: 88.89%
- **Weighted F1-Score**: 87.83%
- **Weighted Precision**: 91.67%
- **Weighted Recall**: 88.89%

### Class-Wise Breakdown
- **Armflapping**: 1.00 Precision | 1.00 Recall | 1.00 F1-Score
- **Spinning**: 0.75 Precision | 1.00 Recall | 0.86 F1-Score
- **Headbanging**: 1.00 Precision | 0.50 Recall | 0.67 F1-Score

---

## ⚠️ Limitations & Disclaimer
AURA is a **preliminary screening aid** designed to alert parents and developmental professionals. It is **not a medical diagnosis tool** and should not replace clinical evaluations by licensed pediatricians or child psychologists.
