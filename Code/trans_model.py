# Use a pipeline as a high-level helper
#from transformers import pipeline

#pipe = pipeline("translation", model="facebook/nllb-200-3.3B")

# Load model directly

from load_dataset import train_ds, val_ds, test_ds
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

def translation(source_lang, target_lang, text):
    #map dataset language code to model language code
    #src_lang= 'eng_Latn'
    #target_lang=
    translator = pipeline('', model=model, tokenizer=tokenizer, src_lang=source_lang, tgt_lang=target_lang, max_length = 400)
    output = translator(text)
    translated_text=output[0]['']
    print(translated_text)


#reference?