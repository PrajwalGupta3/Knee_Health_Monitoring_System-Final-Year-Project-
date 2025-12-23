# OctaKnee: Smart Knee Health Monitoring System
### A Bio-IoT Ecosystem for Real-Time Gait Analysis & GenAI Medical Insights

---

## Overview
**OctaKnee** is an end-to-end **Bio-IoT ecosystem** designed to democratize advanced gait analysis. By integrating wearable IoT sensors with cloud-based Machine Learning and Generative AI, OctaKnee provides patients and doctors with real-time, clinical-grade insights into knee health without the need for expensive lab equipment.

Unlike standard fitness trackers, OctaKnee captures high-fidelity kinematic data, processes it through a custom signal processing pipeline, and uses a **"Defense-in-Depth" Generative AI architecture** to explain the results in plain English.

---

## ⚙️ System Architecture

The system follows a strict **Edge-to-Cloud-to-User** data flow:

1.  **Hardware Layer (Edge):** ESP32 microcontroller + MPU6050 (6-axis IMU) capture raw acceleration and gyroscopic data + BMP180 capture knee temperature.
2.  **Connectivity Layer:** Data is streamed via WiFi/HTTP to **Google Firebase Realtime Database**.
3.  **Processing Layer:** A **Python Flask** backend ingests live data streams.
4.  **Intelligence Layer:**
    * **Signal Processing:** Welch’s Method (Spectral Density) & RMS (Time Domain).
    * **ML Classifier:** Logistic Regression (optimized for <3ms latency).
    * **GenAI Agent:** Llama-3-70B (via Groq) for patient interaction.

![System Architecture Diagram]
<img width="1465" height="818" alt="system architecture" src="https://github.com/user-attachments/assets/0e5e980c-4519-4769-be99-c298cbe43f18" />

---

## Key Innovations

### 1. The "Defense-in-Depth" GenAI Architecture
We moved beyond basic API calls to ensure medical safety:
* **Split-Role Prompting:** Separated "System Instructions" from "User Inputs" to prevent persona drift.
* **Chain-of-Thought (CoT):** Forced the LLM to output internal reasoning inside `<analysis>` tags before generating a final response.
* **Regex Sanitization:** A post-processing layer automatically strips raw CoT logs, ensuring the user only sees clean, empathetic, and verified insights.

### 2. High-Performance ML Pipeline
* **Feature Engineering:** Implemented **Mahalanobis Distance** to mathematically flag outlier data (e.g., stumbles vs. walking).
* **Latency Optimization:** Benchmarked 6 models (SVM, RF, KNN, etc.) and selected **Logistic Regression**, achieving a **2.43µs inference time** for real-time feedback on limited hardware resources.

---

## Tech Stack

| Component | Technology Used |
| :--- | :--- |
| **Hardware** | ESP32, MPU6050  and BMP180 Sensors |
| **Backend** | Python, Flask, Gunicorn |
| **Database** | Google Firebase (Realtime DB) |
| **ML & Signal Proc.** | Scikit-learn, NumPy, SciPy (Welch's Method) |
| **Generative AI** | Llama-3-70B (via Groq API), LangChain concepts |
| **Frontend** | HTML5, CSS3, JavaScript|

---

## Installation & Setup

### Prerequisites
* **Python 3.9+**
* **Git**
* A **Firebase Project** (for Realtime Database)
* A **Groq Cloud API Key** (for Llama-3 inference)

### 1. Clone the Repository
```bash
git clone https://github.com/PrajwalGupta3/Knee_Health_Monitoring_System-Final-Year-Project-.git
cd OctaKnee

### 2. Backend Setup
It is recommended to use a virtual environment to manage dependencies.

Step A: Create Virtual Environment

Bash

# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
Step B: Install Dependencies Install the required Python libraries (Flask, Groq, Pandas, etc.):

Bash

pip install -r requirements.txt
Step C: Configure Environment Variables

Create a file named .env in the root directory.

Copy the contents of .env.example into .env.

Fill in your specific API keys and configuration:

Ini, TOML

# .env file content

# 1. Firebase Configuration
FIREBASE_SERVICE_ACCOUNT=firebase-adminsdk.json
FIREBASE_DB_URL=[https://your-project-id.firebaseio.com](https://your-project-id.firebaseio.com)

# 2. Groq AI Configuration
GROQ_API_KEY=gsk_your_actual_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_API_HOST=[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)

# 3. System Settings
POLL_INTERVAL=10
PORT=5000
Step D: Add Firebase Credentials

Go to your Firebase Console > Project Settings > Service Accounts.

Generate a new private key (JSON file).

Rename this file to firebase-adminsdk.json.

Place it in the root folder of the project (same level as app.py).

Step E: Run the Application Start the Flask server:

Bash

python app.py
The server will start at http://localhost:5000. You should see logs indicating the model loaded successfully.
