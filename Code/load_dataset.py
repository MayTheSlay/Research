#Load AFRIDOC_MT dataset from HuggingFace

from datasets import load_dataset

#max 26k tokens
#ds = load_dataset("masakhane/AfriDocMT", "doc_health") 
train_ds = load_dataset("masakhane/AfriDocMT", "doc_health", split="train")
val_ds= load_dataset("masakhane/AfriDocMT", "doc_health", split="validation")
test_ds= load_dataset("masakhane/AfriDocMT", "doc_health", split="test")


#max 3k tokens
'''''
train_ds = load_dataset("masakhane/AfriDocMT", "doc_health_10", split="train")
val_ds= load_dataset("masakhane/AfriDocMT", "doc_health_10", split="validation")
test_ds= load_dataset("masakhane/AfriDocMT", "doc_health_10", split="test")
'''''


lang_list=list(val_ds.features)
# print(val_ds)
# print(lang_list)
# print(lang_list[1])