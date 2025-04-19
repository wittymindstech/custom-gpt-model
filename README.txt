# Mini GPT Project

### Files:
- `train_mini_gpt.py`: Trains a small GPT-style transformer model and saves it.
- `use_mini_gpt.py`: Loads the trained model and generates text from a prompt.
- `mini_gpt.pth`: Saved model weights.
- `vocab.pkl`: Vocabulary mappings used for encoding/decoding.

### Steps to Run:

1. Train the model:
    `python train_mini_gpt.py`

2. After training, generate text:
    `python use_mini_gpt.py`


You'll be prompted to enter a starting string, and the model will generate text based on it.

Sample Use Case:

`python3 use_mini_gpt.py`
Enter prompt: Capital of China

Generated: Capital of China is Beijina is Beijina is dl o


`python3 use_mini_gpt.py`
Enter prompt: Capital of Delhi

No Output , because Capital of Delhi is not Fed.

