# Load model directly

from load_dataset import train_ds, val_ds, test_ds, lang_list
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re


nllb="facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(nllb)
model = AutoModelForSeq2SeqLM.from_pretrained(nllb)
translator = pipeline('translation', model=model, tokenizer=tokenizer)

#bring amharic punc here
def tokenize_am(text):
    '''''
    AMHARIC PUNC GUIDE:
    ። = full-stop
    ፣ = comma
    ፧ = question mark
    ፤ = semi-colon
    ፥ = colon
    '''''
    punc = re.split(r'\s*(?:።|፡ ፡|፣|፧|፤|፥)\s*', text)
    sentences=[]

    for parts in punc:
        if parts.strip():  # skip empty parts
            sentences.append(parts.strip() + "።")

    return sentences


def translation(source_lang, target_lang, text):
    #map dataset language code to model language code
    language_map={
        'am': 'amh_Ethi',
        'en':'eng_Latn',
        'ha': 'hau_Latn',
        'sw': 'swh_Latn',
        'yo': 'yor_Latn',
        'zu': 'zul_Latn'
    }

    source_code = language_map[source_lang]
    target_code = language_map[target_lang]
    batch_size=5

    # Split into managabel token sizes
    if source_lang == 'am':
        sentences = tokenize_am(" ".join(text))
    else:
        sentences = re.split(r'(?<=[.!?,])\s+', " ".join(text))

    translated_chunks = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        results = translator(batch, src_lang=source_code, tgt_lang=target_code)
        translated_chunks.extend([r['translation_text'] for r in results])

    return " ".join(translated_chunks)

# with open("trans_output.txt", "w", encoding="utf-8") as f:
#     for row in val_ds:
#         #for lang in lang_list:  # lang_list is ['am', 'en', 'ha', 'sw', 'yo', 'zu']
#             text = row['en']

#             translated_text= translation(lang_list[5], lang_list[1], text)

#             f.write(f"Translation for language {lang_list[5]} to {lang_list[1]}:\n")
#             for sentence in translated_text:
#                 f.write(sentence + "\n")
#             f.write("\n---\n\n")

# with open("trans_output.txt", "w", encoding="utf-8") as f:
#     text='ከእናት ወደ ልጅ መተላለፍን ለማስወገድ እንዲረዳ የአለም ጤና ድርጅት ሁሉም ነፍሰ ጡር እናቶች በእርግዝና ወቅት የሄፐታይተስ ቢ ምርመራ እንዲያደርጉ ይመክራል:: የምርመራው ውጤት ፓዘቲቭ ከሆነ ህክምና ሊደረግላቸው እና ለሚወልዶቸው ልጆች ክትባቶች ሊሰጣቸው ይገባል:: ይሁን እንጂ አዲሱ የዓለም ጤና ድርጅት ሪፖርት እንደሚያሳየው ፖሊሲ ካላቸው 64 አገሮች ውስጥ 32 አገሮች ብቻ በቅድመ ወሊድ ክሊኒኮች ውስጥ ሄፓታይተስ ቢን ለመመርመር እና ለመቆጣጠር እንደሚተገብሩ ዘግበዋል:: ሪፖርቱ ከ103 አገሮች ውስጥ 80% የሚሆኑት በኤች አይ ቪ ክሊኒኮች ውስጥ ሄፓታይተስ ቢን ለመመርመር እና ለመቆጣጠር ፖሊሲ እንዳላቸው ያሳያል፤ 65% ደግሞ ለሄፐታይተስ ሲ ተመሳሳይ ፓሊሲ አላቸው:: በኤች አይ ቪ ፕሮግራሞች ውስጥ የሄፐታይተስ ምርመራ እና ህክምና መጨመር ኤችአይቪ ያለባቸውን ሰዎች በጉበት ሲሮሲስ (የጉበት ቆሚ እና የማይመለስጉዳት) እና በጉበት ካንሰር እንዳይያዙ ይከላከላል:: ከዓመታት ወዲህ እየጨመረ የመጣው ሕክምና፤ የሄፐታይተስ ሲ የፈውስ ሕክምናን የሚያገኙ ሰዎች ቁጥር እየጨመረ ነው:: ህክምናን በማስፋፋት ላይ ያለውን እድገት ለማፋጠን የመድሃኒት ዋጋ መቀነስን የአለም ጤና ድርጅት ያበረታታል:: ሄፓታይተስ ሲን ለመፈወስ የ12 ሳምንት የመድሃኒት ህክምና አሁን ዝቅተኛ ገቢ ላላቸው ሀገራት 60 የአሜሪካ ዶላር ያስወጣል፤ ይህም ከፍተኛ ገቢ ባላቸው ሀገራት ለመጀመሪያ ጊዜ ሲገባ ከነበረው ከ90 000 ዶላር በላይ ከነበረው ዋጋ ቀንሷል።'
#     translated_text= translation(lang_list[0], lang_list[1], text)
#     f.write(translated_text)

# print("Translation saved to trans_output.txt")


#reference?
#https://medium.com/@FaridSharaf/text-translation-using-nllb-and-huggingface-tutorial-7e789e0f7816 
