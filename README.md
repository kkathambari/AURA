# 🧠 AURA: Automated Behavioral Screening & Temporal Analysis System



> **AURA** is an AI-powered, computer vision–based behavioral screening system that assists in the preliminary detection of **Motor Stereotyped Behaviors (MSBs)** associated with **Autism Spectrum Disorder (ASD)**. By combining deep spatial feature extraction with temporal sequence modeling, AURA provides an objective, privacy-preserving screening aid for early behavioral assessment.



> ⚠️ **Disclaimer:** AURA is a **preliminary screening tool** and **does not provide a medical diagnosis**. It is intended to assist parents and healthcare professionals and should not replace evaluation by qualified clinicians.



---







# 🚀 Key Features



### 🧠 Spatial-Temporal Deep Learning Pipeline



- ResNet-18 for spatial feature extraction

- Temporal Convolutional Network (TCN) for sequence learning

- Sliding-window temporal inference

- Smart cumulative voting strategy



---



### 🛡 False Positive Prevention



AURA includes multiple validation stages before making a prediction.



- Human Detection (Haar Cascades + HOG Person Detector)

- Screen Recording Detection

- Motion Energy Analysis

- FFT-based Rhythmicity Verification

- Temporal Confidence Validation



---



### 📊 Automated Screening Reports



Generates professional reports including:



- PDF Report

- HTML Summary

- Behavioral Confidence Scores

- Screening Verdict

- Timestamped Analysis



---



### 🔒 Privacy-First Edge AI



Unlike cloud-based systems, AURA performs all processing locally.



- No internet required

- No video upload

- Offline inference

- User-controlled data storage



---



### 🖥 Dual User Interface



- Modern CustomTkinter launcher

- Standard Tkinter analysis dashboard

- Multi-threaded inference

- Responsive UI during video processing



---



# 🎯 Why AURA?



Traditional behavioral screening often depends on manual observation, making the process subjective and time-consuming.



AURA aims to support this process by:



- Objectively measuring repetitive motor behaviors

- Reducing observer variability

- Providing rapid preliminary screening

- Generating structured reports for specialist consultation

- Preserving patient privacy through local processing



---



# 🏗 System Architecture



```text

+------------------------------------------------------------+

|                       USER INTERFACE                       |

|  - desktop_app.py (Tkinter Dashboard)                     |

|  - aura_app.py (CustomTkinter Launcher)                   |

+------------------------------------------------------------+

                        |

                        v

+------------------------------------------------------------+

|                    Worker Thread                           |

|      Multi-threaded processing & progress updates          |

+------------------------------------------------------------+

                        |

                        v

+------------------------------------------------------------+

|                  Inference Engine                          |

|         ResNet18 + Temporal Convolution Network            |

+------------------------------------------------------------+

                        |

                        v

+------------------------------------------------------------+

|            False Positive Prevention Layer                 |

| Human Detection • Motion Energy • FFT • Screen Filter      |

+------------------------------------------------------------+

                        |

                        v

+------------------------------------------------------------+

|                 Report Generation                          |

|              PDF + HTML Screening Reports                  |

+------------------------------------------------------------+

```



---



# 🔄 Complete Workflow



```text

Video Upload

      │

      ▼

Frame Extraction

      │

      ▼

Frame Preprocessing

      │

      ▼

False Positive Prevention

      │

      ▼

ResNet-18 Feature Extraction

      │

      ▼

Temporal Clip Generation

      │

      ▼

Temporal Convolution Network

      │

      ▼

FFT Rhythm Verification

      │

      ▼

Smart Voting Strategy

      │

      ▼

Screening Verdict

      │

      ▼

PDF & HTML Report Generation

```



---



# 🧠 Machine Learning Pipeline



## Spatial Feature Extraction



- Pretrained ResNet-18

- ImageNet Transfer Learning

- 512-dimensional feature vectors

- Feature extraction from every sampled frame



---



## Temporal Sequence Learning



- Temporal Convolutional Network (TCN)

- Dilated Convolutions

- Sliding Windows

- 10-frame temporal clips

- 50% overlap (Stride = 5)



---



## Decision Strategy



Instead of relying on a single prediction,



AURA:



- Evaluates multiple temporal clips

- Aggregates probabilities

- Uses cumulative voting

- Produces a stable screening estimate



---



# 📂 Dataset



The project was trained using a curated subset of the **Self-Stimulatory Behavior Dataset (SSBD)**.



### Dataset Summary



| Property | Value |

|----------|-------|

| Original Dataset | 75 Videos |

| Project Subset | 36 Videos |

| Extracted Temporal Clips | 180 |

| Behaviors | Arm Flapping, Head Banging, Body Spinning |



---



# 📊 Model Performance



## Overall Performance



| Metric | Score |

|---------|-------|

| Accuracy | **88.89%** |

| Precision | **91.67%** |

| Recall | **88.89%** |

| Weighted F1 Score | **87.83%** |



---



## Class-wise Performance



| Behavior | Precision | Recall | F1 Score |

|----------|-----------|--------|----------|

| Arm Flapping | 1.00 | 1.00 | 1.00 |

| Head Banging | 1.00 | 0.50 | 0.67 |

| Body Spinning | 0.75 | 1.00 | 0.86 |



---



# 🛠 Technology Stack



### Programming



- Python



### Deep Learning



- PyTorch

- TorchVision



### Computer Vision



- OpenCV

- Pillow



### GUI



- Tkinter

- CustomTkinter



### Scientific Computing



- NumPy

- Scikit-learn



### Reporting



- FPDF2

- HTML

- CSS



---







# ⚙ Installation



Clone the repository



```bash

git clone https://github.com/kkathambari/AURA.git

cd AURA

```



Create a virtual environment



```bash

python -m venv venv

```



Activate it



### Windows



```bash

venv\Scripts\activate

```



### Linux / macOS



```bash

source venv/bin/activate

```



Install dependencies



```bash

pip install -r requirements.txt

```



---



# ▶ Running the Project



Launch the modern application



```bash

python aura_app.py

```



Launch the expert dashboard



```bash

python desktop_app.py

```



Evaluate the trained model



```bash

python evaluate_model.py

```



---



# 📄 Generated Outputs



AURA automatically generates:



- 📄 PDF Screening Report

- 🌐 HTML Summary Report

- 📊 Behavioral Confidence Statistics

- 📈 Screening Verdict



---



# ⚠ Limitations



- Trained on a relatively small dataset

- Detects only three Motor Stereotyped Behaviors

- Does not analyze facial expressions

- Does not perform gaze tracking

- Not intended for clinical diagnosis

- Requires reasonably clear human-centered video



---



# 🔮 Future Improvements



- Pose Estimation (MediaPipe / YOLO Pose)

- Vision Transformers (ViViT, TimeSformer)

- Audio-based behavioral analysis

- Eye-gaze tracking

- Mobile deployment

- Larger multi-center datasets

- Multi-modal AI screening







---



# 👩‍💻 Developer



**Kathambari K**



Integrated M.Tech – Computer Science & Engineering



AI • Machine Learning • Computer Vision



---



## ⭐ If you found this project interesting, consider giving it a star!
