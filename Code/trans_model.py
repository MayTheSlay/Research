# Use a pipeline as a high-level helper
#from transformers import pipeline

#pipe = pipeline("translation", model="facebook/nllb-200-3.3B")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

def translation(src_lang, target_lang):
    src_lang= eng_Latn
    #target_lang=


