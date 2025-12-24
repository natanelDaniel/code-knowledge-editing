import requests
import json
from time import sleep

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def generate_text_from_codellama(prompt: str) -> str:
    """
    Generates text from the CodeLlama model via Ollama based on a given prompt.

    This version includes 'stop' sequences in the options to prevent the model
    from generating additional conversation history (like subsequent 'Answer:' or 'Comment:').

    Args:
        prompt (str): The input text prompt for the CodeLlama model.

    Returns:
        str: The generated text from the CodeLlama model.

    Raises:
        Exception: For any errors during the API call.
    """
    
    headers = {"Content-Type": "application/json"}
    
    # --- Corrected Data Payload for Ollama ---
    data = {
        "model": "codellama:7b-code",
        "prompt": prompt,
        "stream": False,
        "temperature": 0.1,
        # Ollama parameters for generation control go inside the 'options' object
        "options": {
            # Use 'num_predict' for maximum tokens to generate (1024 is a reasonable default)
            "num_predict": 1024,
            # Critical: These stop sequences instruct the model to immediately halt
            # generation when it encounters these strings, preventing chat continuation.
            "stop": [
                "\end{code}",       # Stops after a single code block is closed
                "Answer:",          # Stops before generating a second "Answer" turn
                "Comment:",         # Stops before generating a user comment
                "Generated text:",  # Stops before starting a new prompt/response pair
            ]
        }
    }
    # ----------------------------------------
    
    # Use exponential backoff for robustness
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # Raise an exception for HTTP errors

            response_data = response.json()
            return response_data.get("response", "")

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                sleep(sleep_time)
            else:
                raise Exception(f"Failed to communicate with Ollama after {max_retries} attempts: {e}")

if __name__ == '__main__':
    # Example usage:
    # Note: This will only work if an Ollama instance is running locally with the model loaded.
    try:
        example_prompt = "Write a Python function to check if a number is prime."
        print(f"Querying Ollama with prompt: '{example_prompt}'")
        generated_code = generate_text_from_codellama(example_prompt)
        print("\n--- Ollama Response (Filtered) ---\n")
        print(generated_code)
        print("\n-----------------------------------\n")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure Ollama is running and the 'codellama:7b-code' model is available.")