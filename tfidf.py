import math

data = [
    "Jane Doe, a 40-year-old female patient, underwent an MRI scan of her brain without contrast, as referred by Dr. Lee. The examination of the brain's tissues (parenchyma), ventricles, and sulci showed no abnormalities. There were no signs of mass, hemorrhage, or acute infarct. The overall impression from the scan indicates that the MRI of the brain is normal.",
    "John Smith, a 55-year-old male, was referred by Dr. Adams for a CT scan of the abdomen. The examination revealed that all abdominal organs were normal, with no signs of tumors, cysts, or stones. The vasculature also appeared normal. The overall impression was that the CT scan of the abdomen was normal.",
    "Emily Davis, a 30-year-old female, underwent a chest X-ray as referred by Dr. Thompson. The examination found clear lungs and a normal heart size, with no evidence of fractures, masses, or pneumonia. Both the diaphragm and rib cage appeared normal. The impression concluded that the chest X-ray was normal, with no signs of disease or injury.",
]

# Tokenizar - Limpiar
tokenizer = [d.lower().split() for d in data]
total_tokens = len(tokenizer)


# Term frequency table
def term_frequency(tokens):
    tf = {}
    for i, v in enumerate(tokens):
        tf[i] = {}
        for t in v:
            tf[i][t] = tf[i].get(t, 0) + 1
    for k, v in tf.items():
        for x, y in v.items():
            tf[k][x] = 1 + math.log(y)
    return tf


# Inverse Document frequency
def idf(tokens, total_tokens):
    idf = {}
    for v in tokens:
        for t in set(v):
            idf[t] = idf.get(t, 0) + 1
    for k, v in idf.items():
        idf[k] = math.log(total_tokens / float(v)) + 1
    return idf


def tfidf(tokens, idf, tf):
    tfidf = {}
    for i, v in enumerate(tokens):
        tfidf[i] = {}
        for t in v:
            tfidf[i][t] = tf[i].get(t, 0) * idf.get(t, 0)
    return tfidf


def tfidf_vector(tokens, tfidf):
    sorted_data = sorted(set([t for d in tokens for t in d]))
    vectors = []
    for i in range(len(tokens)):
        vector = [tfidf[i].get(t, 0) for t in sorted_data]
        vectors.append(vector)
    return vectors


def cosine_similarity(v1, v2):
    dot_product = sum(v1[i] * v2[i] for i in range(len(v1)))
    magnitude_v1 = math.sqrt(sum(v * v for v in v1))
    magnitude_v2 = math.sqrt(sum(v * v for v in v2))
    return dot_product / (magnitude_v1 * magnitude_v2)


def normalize(data_tfidf):
    for k, v in data_tfidf.items():
        magnitude = math.sqrt(sum(y**2 for y in v.values()))
        for x, y in v.items():
            data_tfidf[k][x] = y / magnitude
    return data_tfidf


data_tf = term_frequency(tokenizer)
data_idf = idf(tokenizer, total_tokens)
data_tfidf = tfidf(tokenizer, data_idf, data_tf)
data_tfidf = normalize(data_tfidf)

data_vector = tfidf_vector(tokenizer, data_tfidf)

query = ["X-ray"]
query_tokens = [d.lower().split() for d in query]
query_tf = term_frequency(query_tokens)
query_tfidf = tfidf(query_tokens, data_idf, query_tf)
query_tfidf = normalize(query_tfidf)

scores = []

# save -> data_tfidf, data_idf

query_keys = [v.keys() for _, v in query_tfidf.items()]
for k, v in data_tfidf.items():
    terms = set(query_keys[0]).intersection(set(v.keys()))
    score = sum(query_tfidf[0][term] * v[term] for term in terms)
    scores.append(score)

ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
print(ranked)
