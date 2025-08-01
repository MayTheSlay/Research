#Load AFRIDOC_MT dataset from HuggingFace

from datasets import load_dataset

train_ds = load_dataset("masakhane/AfriDocMT", "doc_health_10", split="train")
val_ds= load_dataset("masakhane/AfriDocMT", "doc_health_10", split="validation")
test_ds= load_dataset("masakhane/AfriDocMT", "doc_health_10", split="test")

print(val_ds)