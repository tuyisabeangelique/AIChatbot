# AI-Powered Restaurant Chatbot

<img width="835" alt="Screenshot 2025-06-20 at 5 55 23 PM" src="https://github.com/user-attachments/assets/c1c82b1d-7c56-45d3-a89e-84af05446225" />


## Overview
This project is an AI-powered chatbot that provides restaurant recommendations. It features a deep learning backend for natural language understanding and a web-based interface with a real-time analytics dashboard to monitor performance.

## Key Features

### AI/ML Capabilities
- **Neural Network Intent Classification**: A model built with TensorFlow/Keras understands user requests.
- **Real-time Analytics Dashboard**: Tracks and visualizes chatbot performance metrics.
- **Natural Language Processing**: Uses NLTK for text preprocessing and analysis.

### Application Features
- **Web-Based Interface**: A clean and modern chat interface built with Flask.
- **Dynamic Responses**: Provides dynamic, simulated data for features like restaurant wait times.
- **Performance Metrics**: The analytics dashboard shows interactions, confidence scores, and response times.

## Technologies Used

### Core AI/ML Stack
- **Python**: Primary development language.
- **TensorFlow (Keras)**: For building the deep learning model.
- **NLTK**: For natural language processing.

### Data Science & Analytics
- **Pandas**: For data manipulation.
- **Matplotlib/Seaborn**: For data visualization.
- **NumPy**: For numerical operations.

### Web Application
- **Flask**: Web framework for the server.
- **HTML/CSS/JavaScript**: For the user interface.

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

### Running the Application

1.  **Train the AI model:**
    ```bash
    python training.py
    ```

2.  **Start the Flask server:**
    ```bash
    python app.py
    ```

3.  **Access the application:**
    - **Chatbot Interface**: [http://127.0.0.1:5000](http://127.0.0.1:5000)
    - **Analytics Dashboard**: [http://127.0.0.1:5000/analytics](http://127.0.0.1:5000/analytics)

