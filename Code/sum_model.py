from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
from load_dataset import lang_list
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

# Load multilingual sentence transformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def tokenize_am(text):
    # Return sentences with full stop ። re-attached
    return [sent.strip() + '።' for sent in text.split('።') if sent.strip()]

def extractive_summary(language, text, clusters):
    # 1. Sentence splitting
    if language == 'am':
        sentences = tokenize_am(text)
    else:
        sentences = sent_tokenize(text)

    # if len(sentences) <= num_sentences:
    #     return ' '.join(sentences)

    # 2. Embedding
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    #doc_embedding = model.encode(text, convert_to_tensor=True)

    #print(sentence_embeddings)

    #https://medium.com/@akankshagupta371/understanding-text-summarization-using-k-means-clustering-6487d5d37255
    # 3. Similarity scoring
    #clusters=1
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

    for i in sorted(my_list):
        print(sentences[i])

    #similarities=cosine_similarity(sentence_embeddings, doc_embedding)



eng_text='Social and behaviour change communication strategies are used to change nutrition-related behaviours. Interventions to address the underlying and basic causes of anaemia look at issues such as disease control, water, sanitation and hygiene, reproductive health and root causes such as poverty, lack of education and gender norms. Anaemia, as a public health issue, needs to be addressed from multiple perspectives and through multiple coordinated efforts, including multiple government sectors, nongovernmental organizations, United Nations agencies and the private sector – each with specific and complementary roles to collectively achieve anaemia reduction and improve health and well-being. See here for more information.'
amh_text='ደም-ማነስ አጠቃላይ እይታ፡ የደም ማነስ የቀይ የደም ሴሎች ቁጥር ወይም በውስጣቸው ያለው የኦክስጅን ተሸካሚ እና የቀይ የደም ሴል አቅላሚ ንጥረ ነገር ( Haemoglobin) ክምችት ከመደበኛው ያነሰበት ሁኔታ ነው። ኦክስጅንን ለመሸከም የኦክስጅን ተሸካሚ እና የቀይ የደም ሴል አቅላሚ ንጥረ ነገር ( Haemoglobin) ያስፈልጋል እና በጣም ጥቂት ወይም ያልተለመደ ቀይ የደም ሕዋሳት ካሉዎት, ወይም በቂ የኦክስጅን ተሸካሚ እና የቀይ የደም ሴል አቅላሚ ንጥረ ነገር ( Haemoglobin) ከሌለዎት, ወደ ሰውነት ህብረህዋሳት ኦክስጅንን ለማጓጓዝ የደሙ አቅም ይቀንሳል. ይህ እንደ ድካም፣ድክመት፣ማዞር እና የትንፋሽ ማጠር የመሳሰሉ ምልክቶችን ያስከትላል። የሰውነት ጤናማ ወይም መደበኛ ተግባር ባህሪ ያለው ( physiologic) ፍላጎቶች ለማሟላት የሚያስፈልገው የየኦክስጅን ተሸካሚ እና የቀይ የደም ሴል አቅላሚ ንጥረ ነገር ( Haemoglobin) መጠን በዕድሜ፣ በጾታ ግንኙነት፣ በመኖሪያ ቤት ከፍታ፣ በማጨስ ልማድና በእርግዝና ደረጃ ይለያያል። የደም ማነስ ምክንያት ሊሆኑ የሚችሉ በርካታ ምክንያቶች ሊኖሩ ይችላሉ ፤ ከእነዚህም መካከል የተመጣጠነ ምግብ እጥረት ወይም በቂ ንጥረነገሮችን አለመጠቀም ፣ ኢንፌክሽኖች (ለምሳሌ የወባ በሽታ ፣ ጥገኛ ኢንፌክሽን ፣ ሳንባ ነቀርሳ ፣ ኤች አይ ቪ)፣ ቁስል ፣ ሥር የሰደዱ በሽታዎች ፣ የማህፀንና የወሊድ ሕመሞች እንዲሁም የወረሱ ቀይ የደም ሴሎች መዛባት ይገኙበታል ። ምንም እንኳ ቫይታሚን B9(folate)፣ በቫይታሚኖች ቢ12 እና በ ኤ ውስጥ ያለው ጉድለት ዋነኛ መንስኤ ዎች ቢሆኑም ለደም ማነስ ዋነኛው ምክንያት የብረት እጥረት ነው። የደም ማነስ በተለይ በትናንሽ ህጻናት፣ በወር አበባ ላይ ያሉ ታዳጊ ልጃገረዶች እና ሴቶች፣ እርጉዝ እና ድህረ ወሊድ ሴቶችን የሚያጠቃ ከባድ የአለም የህዝብ ጤና ችግር ነው። የአለም ጤና ድርጅት ከ6-59 ወር እድሜ ያላቸው 40% ህጻናት፣ 37% ነፍሰ ጡር እናቶች እና 30% ሴቶች ከ15-49 አመት ውስጥ 30% የሚሆኑት የደም ማነስ ችግር አለባቸው ብሏል። የበሽታው ምልክቶች፦ አናሚያ ድካም፣ ማዞር ወይም ቀላል ራስ ምታት፣ እንቅልፍ ማጣትና በተለይ በድካም ጊዜ የትንፋሽ እጥረት ሊያስከትሉ የሚችሉ የተለያዩ ለየት ያሉ ምልክቶችን ሊያስከትል ይችላል። በተለይ ህጻናት እና ነፍሰ ጡር እናቶች ለአደጋ የተጋለጡ ሲሆኑ የደም ማነስ ችግር በከፋ ሁኔታ የእናቶች እና የህጻናት ሞት የመጋለጥ እድልን ይጨምራል። በተጨማሪም የብረት እጥረት በህፃናት የአዕምሮ እና የአካል እድገት ላይ ተፅዕኖ እንደሚያሳድርና በአዋቂዎች ላይ ያለውን ምርታማነት እንደሚቀንስ ተገልጿል። የደም ማነስ የሁለቱም የተመጣጠነ ምግብ እጥረት እና የጤና መጓደል አመላካች ነው። በራሱ ችግር ያለበት ነው፣ ነገር ግን የአካል ብቃት እንቅስቃሴ ጉልበት በማጣት የተነሳ ሌሎች የአለም አቀፍ የህዝብ ጤና ስጋቶችን እንደ መቀንጨር እና መመናመን፣ ዝቅተኛ የልደት ክብደት እና የልጅነት ከመጠን ያለፈ ውፍረት ሊመጣ ይችላል። በህፃናት የትምህርት ውጤት እና በአዋቂዎች ላይ የሚደርሰው በደም ማነስ ምክንያት የስራ ምርታማነት መቀነስ በግለሰቡእና በቤተሰብ ላይ ተጨማሪ ማህበራዊና ኢኮኖሚያዊ ተፅእኖ ሊያሳድር ይችላል። የዓለም ጤና ድርጅት ምላሽ ደም ማነስ የዚህን ችግር ሸክምና በተወሰነ ህዝብ ውስጥ የጤና እና የበሽታ መከሰት እና ስርጭትን የሚወስኑ ጥናትን (epidemiology) ለመረዳት፣ የሕዝብ ጤና ጣልቃ ገብነት ለማድረግና በሕይወት ዘመኑ ሁሉ ሰዎችን የሕክምና ክትትል ለማድረግ በጣም ወሳኝ ነው። የብረት እጥረት በጣም የተለመደ ና ብዙውን ጊዜ በአመጋገብ ለውጥ አማካኝነት ሊታከም የሚችል ቢሆንም ከበስተጀርባ ያሉ ኢንፌክሽኖችንና የተሟላ የጤና መታወክ የሚጠይቁ ሥር የሰደዱ ችግሮችን በመፍታት ሌሎች ደም ማነስ ዓይነቶችን ማከም ያስፈልጋል። በመከላከል እና በህክምና የደም ማነስን ስርጭት ለመቀነስ የሚረዳ ሁሉንም የአለም ጤና ድርጅት ክልሎችን የሚሸፍን መመሪያ አለው። እነዚህ መመሪያዎች የአመጋገብ ልዩነትን ለመጨመር፣ የጨቅላ ህፃናት አመጋገብን ለማሻሻል እና ጥቃቅን ንጥረ ነገሮችን የመዳኒት ወይም ሌላ ንጥረ ነገር በሰውነት ውስጥ የመዋጥ እና የመጠቀም ችሎታ (Bioavipability) እና አወሳሰድን በብረት፣ ፎሊክ አሲድ እና ሌሎች ቪታሚኖች እና ማዕድኖችን በማጠናከር ወይም በመመገብ ማሻሻል ያለመ ነው። ከአመጋገብ ጋር የተያያዙ ባህሪያትን ለመለወጥ ማህበራዊ እና የባህርይ ለውጥ የግንኙነት ስልቶች ጥቅም ላይ ይውላሉ። የደም ማነስን ዋና እና መሰረታዊ መንስኤዎችን ለመፍታት የተደረገው ጣልቃገብነት እንደ በሽታን መቆጣጠር፣ ውሃ፣ የአካባቢ ንፅህና እና የራስን ንፅህና፣ የስነ ተዋልዶ ጤና እና ሥር ነቀል እንደ ድህነት፣ የትምህርት እጥረት እና የስርዓተ-ፆታ ደንቦችን የመሳሰሉ ጉዳዮችን ይመለከታል። ደም ማነስ የሕዝብ ጤና ጉዳይ እንደመሆኑ መጠን ከተለያየ አቅጣጫና በተለያዩ የተቀናጀ ጥረቶች፣ በርካታ የመንግሥት ዘርፎችን፣ መንግስታዊ ያልሆኑ ድርጅቶችን፣ የተባበሩት መንግስታት ድርጅትን እና የግል ዘርፉን ጨምሮ፣ እያንዳንዳቸው ደም ማነስን ለመቀነስ እና ጤናን እና ደህንነትን ለማሻሻል የተወሰኑ እና ማሟያ ዎች ያሏቸው መሆን አለባቸው። ለበለጠ መረጃ እዚህ ላይ ይመልከቱ።'
#text='Anaemia, as a public health issue, needs to be addressed from multiple perspectives and through multiple coordinated efforts, including multiple government sectors, nongovernmental organizations, United Nations agencies and the private sector – each with specific and complementary roles to collectively achieve anaemia reduction and improve health and well-being.'
print(extractive_summary(lang_list[0], amh_text, 1))