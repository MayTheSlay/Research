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
    punc = re.split(r'\s*(?:።|፡ ፡|፧)\s*', text)
    sentences=[]

    for parts in punc:
        if parts.strip():  # skip empty parts
            sentences.append(parts.strip() + "።")

    return sentences

punkt_params = PunktParameters()
punkt_params.abbrev_types = set(['e.g', 'i.e', 'etc', 'Dr', 'Mr', 'Ms', '1', '2', '3', '4', '5'])
tokenizer = PunktSentenceTokenizer(punkt_params)

def tokenize(text):
    # 1. Insert special marker after sentence-ending punctuation with quotes
    text = re.sub(r'([.!?][”"])', r'\1 <SPLIT>', text)

    # 2. First pass: use Punkt tokenizer
    sentences = tokenizer.tokenize(text)

    # 3. Second pass: split by our <SPLIT> marker
    result = []
    for sent in sentences:
        parts = [p.strip() for p in sent.split("<SPLIT>") if p.strip()]
        result.extend(parts)

    return result

def extractive_summary(language, text, clusters):
    # 1. Sentence splitting
    if language == 'am':
        sentences = tokenize_am(text)
    else:
        sentences = tokenize(text)

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

file_path=r"C:\Users\mayur\OneDrive\Desktop\Research\Code\base_summaries"
lang=lang_list[0]
with open(f"{file_path}/{lang}_sum.txt", "w", encoding="utf-8") as f:
    for i, row in enumerate(val_ds):
        #for lang in lang_list:  # lang_list is ['am', 'en', 'ha', 'sw', 'yo', 'zu']
        
            text = row[lang]

            summaries= extractive_summary(lang, text, 5)
            f.write(f"Document {i+1}\n")

            for sentence in summaries:
                f.write(sentence + "\n")
            
            f.write("\n---\n\n")

print(f"Summaries saved to base summaries/{lang}_sum.txt")
