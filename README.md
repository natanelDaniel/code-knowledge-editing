# Ollama Quickstart: Basic Text Generation with CodeLlama

This project demonstrates a basic integration with the CodeLlama Large Language Model (LLM) via Ollama for text generation.

## Installation

To get started with this project, follow these steps:

1.  **Install Ollama:**
    Download and install Ollama from [ollama.com](https://ollama.com/download).

2.  **Download the `codellama:7b` model:**
    Open your terminal and run:
    ```bash
    ollama pull codellama:7b
    ```

3.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    ```

4.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

5.  **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **IMPORTANT: Ensure Ollama server is running.**
    The Ollama application must be running in the background for the API to be accessible. If it's not running, please start it.

2.  **Run the main script:**
    ```bash
    python main.py
    ```

## Troubleshooting

*   **`Error: Could not connect to Ollama server.`**
    This error indicates that the Ollama application is not running or is not accessible. Please ensure Ollama is installed and running on your machine. You can typically start Ollama by launching the application or running `ollama serve` in your terminal.

*   **`404 Client Error: Not Found for url: http://localhost:11434/api/generate`**
    This error usually means that the `codellama:7b` model is not available or not loaded in your Ollama instance. Please ensure you have pulled the model. Open your terminal and run:
    ```bash
    ollama list
    ```
    If `codellama:7b` is not listed, pull it using:
    ```bash
    ollama pull codellama:7b
    ```
    After pulling the model, try running `main.py` again.
