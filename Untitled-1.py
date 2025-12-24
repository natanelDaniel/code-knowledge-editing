from easyeditor import MENDHyperParams, BaseEditor

# -----------------------------------------------------------
# Step 1 & 2: Define Model Configuration and Editing Method
# -----------------------------------------------------------

## We use the MEND method here, so we import `MENDHyperParams`.
## This file must exist in your EasyEdit hparams directory.
hparams = MENDHyperParams.from_hparams('./hparams/MEND/gpt2-xl')

# -----------------------------------------------------------
# Step 3: Define Edit Descriptor (The NumPy Code Edit)
# -----------------------------------------------------------

## We define the prompts to be edited. The edit targets the removal of 'np.tostring'
## in favor of 'np.tobytes', as mandated by NumPy 2.3.0 release notes.
prompts = [
    'How do I convert a NumPy array "arr" to a string of bytes using the deprecated method?',
]

## Ground truth: The expected output BEFORE the edit (the deprecated function).
ground_truth = [
    'arr.tostring()',
]

## Edit target: The expected output AFTER the edit (the modern, correct function).
target_new = [
    'arr.tobytes()',
]

# -----------------------------------------------------------
# Step 4: Initialize the Editor
# -----------------------------------------------------------

## EasyEdit provides a simple and unified way to initialize the Editor.
editor = BaseEditor.from_hparams(hparams)

# -----------------------------------------------------------
# Step 5: Provide Data for Locality Evaluation (Preservation Check)
# -----------------------------------------------------------

## Locality data checks whether unrelated knowledge or code is preserved.
locality_inputs = {
    'preserved_code':{
        # Check an essential, unchanged API call (e.g., np.unique)
        'prompt': ['What function is used to find unique values in a NumPy array?',
         'What is the standard function for matrix multiplication in NumPy?'],
        'ground_truth': ['np.unique', 'np.dot']
    },
}

# -----------------------------------------------------------
# Step 6: Execute the Edit and Evaluation
# -----------------------------------------------------------

## The edit function returns a series of metrics and the modified model weights.
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    locality_inputs=locality_inputs,
    sequential_edit=False # Set True for Continuous Editing
)

## metrics: includes edit success (rewrite_acc), generalization, and locality scores.
## edited_model: the language model with the updated code knowledge.