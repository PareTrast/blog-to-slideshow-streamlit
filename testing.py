import os
import torch
from transformers import pipeline, AutoTokenizer

# Apply the torch patch again just in case (for completeness)
torch.classes.__path__ = []

MODEL_NAME = "google/pegasus-cnn_dailymail"

print("Attempting to load tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded successfully!")

print("Attempting to load summarization pipeline...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
summarizer_pipeline = pipeline(
    "summarization",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    device=device
)
print("Summarization pipeline loaded successfully!")

# Try a simple summarization test
test_text = "The quick brown fox jumps over the lazy dog. This sentence contains all letters of the English alphabet. It is often used for testing typewriters and computer fonts. This is a longer text to ensure adequate input."
print("Attempting to summarize a test text...")
test_summary = summarizer_pipeline(test_text, max_length=50, min_length=10, do_sample=False)
print("Test summary generated!")
print(test_summary[0]['summary_text'])

exit() # Type this to exit the interpreter