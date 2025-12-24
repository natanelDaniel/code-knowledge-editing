import json
from EasyEdit.easyeditor import BaseEditor, ROMEHyperParams
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model and method
model_name = 'gpt2-xl'
model_class = 'GPT2LMHeadModel'
tokenizer_class = 'GPT2Tokenizer'
tokenizer_name = 'gpt2-xl'
model_parallel = False # true for multi-GPU editing
method_name = 'ROME'

# Data from extracted_deprecations.json
data = {
    "name": "Removal of np.nbytes",
    "question": "How can I find the size in bytes of a single element of a NumPy array?",
    "old_approach": "The np.nbytes() function returns the size in bytes of a single element.",
    "new_approach": "Use np.dtype(<dtype>).itemsize to get the size in bytes of a single element."
}

# Extract relevant information for editing
prompt = data["question"]
ground_truth = data["old_approach"]
target_new = data["new_approach"]

# Load hyperparameters for ROME
# You'll need to make sure the hparams file exists for llama-2-7b.
# If not, you might need to adapt from an existing one or create it.
# Assuming a hparams file for llama-2-7b ROME exists at 'hparams/ROME/llama-2-7b.yaml'
# You may need to replace this path with the correct path to your hparams file if it's different.
try:
    # Adjust this path if the EasyEdit hparams are not directly accessible
    # A common path after pip install might be something like:
    # C:/Users/21dan/AppData/Local/Programs/Python/Python310/Lib/site-packages/easyeditor/hparams/ROME/llama-2-7b.yaml

    hparams = ROMEHyperParams.from_hparams(f'./hparams/{method_name}/{model_name}')
except FileNotFoundError:
    print(f"Error: Hyperparameters file not found for {model_name} with {method_name}.")
    print("Please ensure the hparams file exists at the specified path, e.g., 'EasyEdit/hparams/ROME/llama-2-7b.yaml'.")
    print("If EasyEdit was installed via pip, the hparams files are usually in the package's installation directory.")
    print("You may need to manually locate them or copy them to your current working directory.")
    exit()

# Construct Language Model Editor
editor = BaseEditor.from_hparams(hparams)

# Perform the edit
print(f"Attempting to edit the model '{model_name}' using '{method_name}' method.")
print(f"Prompt: {prompt}")
print(f"Old Approach (Ground Truth): {ground_truth}")
print(f"New Approach (Target New): {target_new}")

metrics, edited_model, _ = editor.edit(
    prompts=[prompt],
    ground_truth=[ground_truth],
    target_new=[target_new],
    sequential_edit=False
)

print("Editing Metrics:")
print(json.dumps(metrics, indent=4))

# Example of how to use the `edited_model` for inference
# This part requires the model to be loaded on a device and might require authentication
# for Llama models from Hugging Face.
try:
    print("\nTesting edited model with the prompt:")
    print(f"Prompt: {prompt}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure a pad_token is set for the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Or a different appropriate token

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(edited_model.device)
    output = edited_model.generate(input_ids, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
    print("Output after edit:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
except Exception as e:
    print(f"Could not perform inference with the edited model. Error: {e}")
    print("This might be due to missing Hugging Face authentication for Llama models, or insufficient GPU resources.")
    print("Please ensure you have authenticated with Hugging Face if using gated models like Llama.")
    print("You can do this by running `huggingface-cli login` in your terminal.")
