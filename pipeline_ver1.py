# Assuming you have the following setup elsewhere in your script:
# import google.generativeai as genai
# genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
# model = genai.GenerativeModel('gemini-2.5-flash') # Recommended for structured tasks

import requests
import os
# os.environ["GEMINI_API_KEY"] = "AIzaSyAtZ89chzoqrqdxP5VI6-E1_lz4BeTXR8A"
os.environ["GEMINI_API_KEY"] = "AIzaSyCT7CLrY8O1Rfakma8hDSGOVJOKgdRJmp4"
# AIzaSyAG__gP57xvOLrIzFg12E9BscxGW1FtedU
# AIzaSyAtZ89chzoqrqdxP5VI6-E1_lz4BeTXR8A
import json
import google.generativeai as genai # Added import
import google.generativeai.types as types # Added import
import io
import sys
import numpy as np
import time

def run_code_snippet(code_str):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_stdout = io.StringIO()
    redirected_stderr = io.StringIO()
    sys.stdout = redirected_stdout
    sys.stderr = redirected_stderr
    
    returncode = 0
    try:
        exec(code_str, globals())
    except Exception as e:
        sys.stderr.write(str(e))
        returncode = 1 # Indicate an error occurred
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return returncode, redirected_stdout.getvalue(), redirected_stderr.getvalue()

# --- Configuration (Move this out of the function in a real script) ---
# Check for API key in environment variables
if "GEMINI_API_KEY" not in os.environ:
    print("WARNING: GEMINI_API_KEY not found in environment variables. Using placeholder.")
    # Set a dummy key to prevent immediate configuration failure if needed, 
    # but the API call will still fail if a real key isn't used.
    # os.environ["GEMINI_API_KEY"] = "DUMMY_KEY" 

try:
    # Initialize the client (will use the environment variable if set)
    genai.configure() 
    # Use a model that supports structured output (like gemini-2.5-flash)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
except Exception as e:
    print(f"Error configuring Gemini client: {e}")
    gemini_model = None # Set to None if configuration fails
# --- End of Configuration ---

def fetch_release_notes(url):
    """Fetches the content of a given URL."""
    # ... (Function body remains the same)
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def extract_deprecations_with_gemini(release_notes_content):
    """
    Uses Gemini AI to extract deprecated functions and their replacements, 
    enforcing a JSON output structure.
    """
    if gemini_model is None:
        pass
    print("\n--- Calling Gemini AI ---\n")

    # 1. Define the desired JSON structure using a Pydantic schema (or dict)
    # The output should be a list of objects, so the schema must reflect this.
    json_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": """A brief name for the change or
                         deprecation (e.g., 'Removal of array.tostring()')."""},
                "question": {"type": "string",
                             "description": """A focused user question that not
                              explicitly mentions the deprecated function,
                               macro, or module, but mention the reason to use
                                the function. you must not mention that the
                                 function is deparcted or removed or no longer
                                  allowed. just ask about how to somthing that
                                   the answer will be the function (e.g.,
                                    'How can I convert a NumPy array's data
                                     buffer into a standard Python bytes
                                      object?')."""},
                "question_ver2": {"type": "string",
                                "description": """An alternative phrasing of the user question, using different words but conveying the same meaning as 'question'. The answer should remain the same. (e.g., 'What's the method to transform a NumPy array's data buffer into a standard Python bytes object?')."""},
                "old_approach": {"type": "string",
                                 "description": """Provide a descriptive, code explanation of the
                                    deprecated approach, mentioning the deprecated name or function
                                    (up to one line of code) but without mention that its deparcted
                                    (or previously used or was used) - just put the approch itself
                                        (e.g np.tostring). **Write this statement in the present tense**,
                                        as if the problem is the current state"""},
                "new_approach": {"type": "string",
                                 "description": """A concise, plain-language
                                  description of the recommended alternative
                                   function or method (by name) and its benefit.
                                    - just the approch itself. (e.g np.tobytes()
                                     (e.g., 'array.tobytes()"""},
                "subject": {
                    "type": "string",
                    "description": """The specific entity or technical term
                     being modified. CRITICAL: This string MUST appear exactly
                      as a substring within the 'question' field to avoid
                       assertion errors during editing."""
                },
                "completion_prompt": {"type": "string",
                    "description": """A prompt that guides the model in generating a completion based on the provided context. (e.g To convert a NumPy array's data
                                     buffer into a standard Python bytes
                                      object use )"""}
            },
             "required": ["name", "question", "question_ver2", "old_approach", "new_approach", "subject", "completion_prompt"]
        }
    }

    # NOTE: Ensure the variable 'release_notes_content' is populated 
    # with the actual release notes text before generating the prompt.


    prompt = f"""
    Analyze the following release notes content and extract a list of the most
    importent knowledge-focused Q&A objects that describe the changes and
     deprecations - take only that have influence on common code.

    You must adhere to the following strict requirements:
    1.  **question**: Must be a natural user question that not explicitly names
    the function, method, or module involved but mention the reason to use the
     function. you must not mention that the function is deparcted or removed or
      no longer allowed. just ask about how to somthing that the answer will be
       the function (e.g., 'How can I convert a NumPy array's data buffer into a
        standard Python bytes object?'). Do not include runnable code snippets
         do not use deparcted in your question.
    2.  **question_ver2**: An alternative phrasing of the user question, using different words but conveying the same meaning as 'question'. The answer should remain the same. (e.g., 'What's the method to transform a NumPy array's data buffer into a standard Python bytes object?').
    3.  **old_approach**: Provide a descriptive, code explanation of the
     deprecated approach, mentioning the deprecated name or function
      (up to one line of code) but without mention that its deparcted
       (or previously used or was used) - just put the approch itself
        (e.g np.tostring). **Write this statement in the present tense**,
        as if the problem is the current state - do not use : is used, previously. use the words - use, currently (present).
    4.  **new_approach**: Provide a descriptive, non-code explanation of the
     modern, recommended alternative, mentioning the new or function (up to one
      line of code) - just the approch itself. (e.g np.tobytes())
    5. **Subject**: The specific technical entity or object being discussed.
     This MUST be a word or phrase that appears exactly, character-for-character,
      inside the 'question' field.
    6. **completion_prompt**: A prompt that guides the model in generating a completion based on the provided context. (e.g To convert a NumPy array's data
                                     buffer into a standard Python bytes
                                      object use )

    Ensure the output is a **valid JSON array** that strictly conforms to the
     requested schema. **Do not include any text, prose, or code outside the
      JSON block.**

    Release Notes Content:
    {release_notes_content}
    """

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": json_schema,
            },
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return []

if __name__ == "__main__":
    # Ensure the API key is set in your environment variables for this part to work
    # E.g., export GEMINI_API_KEY="AIza..."

    # Example URL for demonstration if user doesn't provide one
    # Note: The actual content fetch will only work with a real, accessible URL.
    urls = ["https://numpy.org/doc/stable/release/2.3.0-notes.html", 
            "https://numpy.org/doc/stable/release/2.2.0-notes.html", 
            "https://numpy.org/doc/stable/release/2.1.0-notes.html", 
            "https://numpy.org/doc/stable/release/2.0.0-notes.html",
            "https://pandas.pydata.org/docs/whatsnew/v2.3.0.html",
            "https://pandas.pydata.org/docs/whatsnew/v2.2.0.html",
            "https://github.com/pytorch/pytorch/releases"
            ]
    release_note_url = "https://numpy.org/doc/stable/release/2.3.0-notes.html"
    # release_note_url = input(f"Enter the URL of the release notes (default: {default_url}): ") or default_url
    
    print(f"Fetching content from {release_note_url}...")
    content = fetch_release_notes(release_note_url)
    
    all_deprecations = []

    for url in urls:
        print(f"\n--- Processing URL: {url} ---\n")
        content = fetch_release_notes(url)
        
        if content:
            print("Content fetched successfully. Extracting deprecations...")
            deprecations = extract_deprecations_with_gemini(content)
            
            if deprecations:
                all_deprecations.extend(deprecations)
                print("\n--- Extracted Deprecations (Q&A Format) for Current URL ---\n")
                for i, dep in enumerate(deprecations):
                    print(f"Deprecation {i+1}:")
                    print(f"  Name: {dep.get('name', 'N/A')}")
                    print(f"  Question: {dep.get('question', 'N/A')}")
                    print(f"  Old Approach: {dep.get('old_approach', 'N/A')}")
                    print(f"  New Approach: {dep.get('new_approach', 'N/A')}")
                    print("-"*30)
                print("\n--- End of Extracted Deprecations for Current URL ---")
            else:
                print("No deprecations extracted or Gemini API call failed for this URL.")
        else:
            print(f"Failed to fetch release notes content from {url}. Skipping.")

        # Add a 1-minute delay before processing the next URL to avoid rate-limiting
        # time.sleep(60)

    if all_deprecations:
        print("\n--- All Extracted Deprecations ---\n")
        # Now, save all_deprecations to a file
        output_file = "extracted_deprecations_v1.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_deprecations, f, indent=4, ensure_ascii=False)
        print(f"Successfully extracted and saved {len(all_deprecations)} deprecations to {output_file}")
    else:
        print("No deprecations were extracted from any URL. Exiting.")
