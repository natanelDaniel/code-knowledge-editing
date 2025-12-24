from api_client import generate_text_from_codellama

if __name__ == "__main__":
    # Define a sample prompt
    sample_prompt = "Write me a function that outputs the fibonacci sequence"

    print(f"Sending prompt: '{sample_prompt}'")

    # Call the text generation function
    generated_text = generate_text_from_codellama(sample_prompt)

    # Print the result
    if generated_text:
        print("\nGenerated text:")
        print(generated_text)
    else:
        print("\nText generation failed or returned empty.")
