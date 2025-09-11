from sum_model import extractive_summary
from trans_model import translation
from load_dataset import val_ds, lang_list

#pipeline number 1
def summarize_then_translate(source_lang, target_lang, text, clusters):

    summary=extractive_summary(source_lang, text, clusters)
    translated=translation(source_lang, target_lang, summary)

    return translated

#pipeline number 2
def translate_then_summarize(source_lang, target_lang, text, clusters):

    translated=translation(source_lang, target_lang, text)
    summary=extractive_summary(target_lang, translated, clusters)

    return summary

with open("am_sum.txt", "w", encoding="utf-8") as f:
    for row in val_ds:
        #for lang in lang_list:  # lang_list is ['am', 'en', 'ha', 'sw', 'yo', 'zu']
            text = row['am']

            summaries= extractive_summary(lang_list[0], text, 1)

            f.write(f"Summary for language={lang_list[0]}:\n")
            for sentence in summaries:
                f.write(sentence + "\n")
            f.write("\n---\n\n")

print("Summaries saved to am_sum.txt")