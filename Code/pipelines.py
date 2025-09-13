from sum_model import extractive_summary
from trans_model import translation
from load_dataset import val_ds, lang_list

#pipeline number 1
def summarize_then_translate(source_lang, target_lang, text, clusters):

    summary=extractive_summary(source_lang, text, clusters)
    print("Summary:", summary)
    translated=translation(source_lang, target_lang, summary)

    return translated

#pipeline number 2
def translate_then_summarize(source_lang, target_lang, text, clusters):

    translated=translation(source_lang, target_lang, text)
    summary=extractive_summary(target_lang, translated, clusters)

    return summary


file_path=r"C:\Users\mayur\OneDrive\Desktop\Research\Code\SumTrans_results"
source='en'
target=lang_list[5]
with open(f"{file_path}/eng_to_{target}_sum.txt", "w", encoding="utf-8") as f:
    for i, row in enumerate(val_ds):
        #for lang in lang_list:  # lang_list is ['am', 'en', 'ha', 'sw', 'yo', 'zu']

        print(f"Translating doc {i+1}")
        text = row[source]

        summaries=summarize_then_translate(source, target, text, 5)
        f.write(f"Document {i+1}\n")

        f.write(summaries + "\n")
            
        f.write("\n---\n\n")

print(f"Summaries saved to sumtrans summaries/{target}_sum.txt")