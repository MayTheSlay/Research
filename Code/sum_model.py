from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
from load_dataset import lang_list, val_ds
from sklearn.cluster import KMeans
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import re
#https://medium.com/@akankshagupta371/understanding-text-summarization-using-k-means-clustering-6487d5d37255 for kmeans 

# Load multilingual sentence transformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def tokenize_am(text):
    # Return sentences with full stop ። re-attached
    punc = re.split(r'\s*(?:።|፡ ፡)\s*', text)
    sentences=[]
    # for sentence in text.split('።') or text.split('::'):
    #     if sentence.strip():
    #         sentences.append(sentence+'።')

    for parts in punc:
        if parts.strip():  # skip empty parts
            sentences.append(parts.strip() + "።")

    return sentences

def tokenize(text):
    punkt_params = PunktParameters()
    punkt_params.abbrev_types = set(['e.g'])

    tokenizer = PunktSentenceTokenizer(punkt_params)
    return tokenizer.tokenize(text)

def extractive_summary(language, text, clusters):
    # 1. Sentence splitting
    if language == 'am':
        sentences = tokenize_am(text)
    else:
        sentences = tokenize(text)

    # if len(sentences) <= num_sentences:
    #     return ' '.join(sentences)

    # 2. Embedding
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    #3 Choosing closest sentences to cluster centres for summary
    means= KMeans(n_clusters=clusters, random_state=42)
    kmeans=means.fit(sentence_embeddings)

    my_list=[]
    for i in range(clusters):
        my_dict={}

        for j in range(len(sentences)):
            if kmeans.labels_[j]==i:
                my_dict[j]=distance.euclidean(kmeans.cluster_centers_[i], sentence_embeddings[j])

        min_distance=min(my_dict.values())
        my_list.append(min(my_dict, key=my_dict.get))

    summaries=[]
    for i in sorted(my_list):
        summaries.append(sentences[i])

    return summaries


#Save to a .txt file
'''''
with open("output.txt", "w", encoding="utf-8") as f:
    for row in val_ds:
        for lang in lang_list:  # lang_list should be ['am', 'en', 'ha', 'sw', 'yo', 'zu']
            text = row[lang]

            # Skip empty text
            # if not text.strip():
            #     continue

            # Get summary for this language's text
            summary_sentences = extractive_summary(lang, text, clusters=3)

            # Save to file
            f.write(f"Summary for language={lang}:\n")
            for sent in summary_sentences:
                f.write(sent + "\n")
            f.write("\n---\n\n")

'''''


with open("eng_sum.txt", "w", encoding="utf-8") as f:
    for row in val_ds:
        #for lang in lang_list:  # lang_list is ['am', 'en', 'ha', 'sw', 'yo', 'zu']
            text = row['en']

            summaries= extractive_summary(lang_list[1], text, 1)

            f.write(f"Summary for language={lang_list[1]}:\n")
            for sentence in summaries:
                f.write(sentence + "\n")
            f.write("\n---\n\n")

print("Summaries saved to eng_sum.txt")

#extractive_summary(lang_list[0], amh_text2, 1)
#print(amh_text)