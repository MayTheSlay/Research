# Use a pipeline as a high-level helper
#from transformers import pipeline

#pipe = pipeline("translation", model="facebook/nllb-200-3.3B")

# Load model directly

from load_dataset import train_ds, val_ds, test_ds
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


nllb="facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(nllb)
model = AutoModelForSeq2SeqLM.from_pretrained(nllb)


def translation(source_lang, target_lang, text):
    #map dataset language code to model language code

    #source_lang= 'yor_Latn'
    #target_lang='eng_latn'
    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_lang, tgt_lang=target_lang, max_length = 400)
    output = translator(text)
    translated_text=output[0]['translation_text']
    print(translated_text)

text='Imbangela evame kakhulu ye-anaemia ngokuphathelene nomsoco, nakuba ukuntuleka kwe-folate, amavithamini B12 no-A nawo ayimbangela ebalulekile.'
translation('zul_Latn', 'eng_Latn', text)
#reference?
#https://medium.com/@FaridSharaf/text-translation-using-nllb-and-huggingface-tutorial-7e789e0f7816 