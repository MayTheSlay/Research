from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import stanza
from load_dataset import train_ds, val_ds, test_ds

#using sentence transformer to embed sentences
embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

#using stanza for tokenization
for languages in val_ds.features:
    #print(languages)
    stanza.download(languages, package="craft")
    pipeline = stanza.Pipeline('en', package='craft')

#def summarization():
