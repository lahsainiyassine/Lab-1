import os
import string
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ============================================================
# TÂCHE 1 — Chargement du corpus
# ============================================================
def load_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

# ============================================================
# TÂCHE 2 — Segmentation en phrases
# ============================================================
def split_sentences(text: str):
    return sent_tokenize(text)

# ============================================================
# TÂCHE 3 — Segmentation en mots
# ============================================================
def tokenize_words(sentences):
    return [word_tokenize(s) for s in sentences]

# ============================================================
# TÂCHE 4 — Suppression des stop words
# ============================================================
STOP_EN = set(stopwords.words('english'))
STOP_FR = set(stopwords.words('french'))

def drop_stopwords(tokenized, lang="en"):
    stop = STOP_EN if lang == "en" else STOP_FR
    return [[w for w in sent if w.lower() not in stop] for sent in tokenized]

# ============================================================
# TÂCHE 5 — Suppression de la ponctuation
# ============================================================
PUNCT = set(string.punctuation)

def drop_punctuation(tokenized):
    return [[w for w in sent if w not in PUNCT] for sent in tokenized]

# ============================================================
# TÂCHE 6 — Normalisation en minuscules + exceptions
# ============================================================
EXCEPT_UPPER = {"NLP", "DATA", "USA"}

def to_lower_with_exceptions(tokenized):
    out = []
    for sent in tokenized:
        row = [w if w in EXCEPT_UPPER else w.lower() for w in sent]
        out.append(row)
    return out

# ============================================================
# TÂCHE 7 — Racinisation vs Lemmatisation
# ============================================================
stemmer = PorterStemmer()
lemmat = WordNetLemmatizer()

def stem(tokenized):
    return [[stemmer.stem(w) for w in sent] for sent in tokenized]

def lemmatize(tokenized):
    return [[lemmat.lemmatize(w) for w in sent] for sent in tokenized]

def flatten(list_of_lists):
    for sent in list_of_lists:
        for w in sent:
            yield w

# ============================================================
# TÂCHE 8.1 — spaCy vs NLTK
# ============================================================
def spacy_tokenize_compare(text: str):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    spacy_sents = [s.text.strip() for s in doc.sents]
    spacy_tokens = [t.text for t in doc if not t.is_space]
    return spacy_sents, spacy_tokens

# ============================================================
# TÂCHE 8.2 — Mesure d'impact sur TF-IDF
# ============================================================
def compare_tfidf_impact(raw_corpus: list, clean_corpus: list, top_k=5):
    vec_raw = TfidfVectorizer()
    vec_clean = TfidfVectorizer()
    
    X_raw = vec_raw.fit_transform(raw_corpus)
    X_clean = vec_clean.fit_transform(clean_corpus)
    
    raw_scores = sorted(zip(vec_raw.get_feature_names_out(), X_raw.toarray().sum(axis=0)), key=lambda x: x[1], reverse=True)
    clean_scores = sorted(zip(vec_clean.get_feature_names_out(), X_clean.toarray().sum(axis=0)), key=lambda x: x[1], reverse=True)
    
    return raw_scores[:top_k], clean_scores[:top_k]

# ============================================================
# TÂCHE 8.3 — Export des paramètres JSON
# ============================================================
def save_config(filepath="preprocess_config.json", lang="en"):
    config = {
        "language": lang,
        "exceptions_upper": list(EXCEPT_UPPER),
        "nb_stopwords_ref": len(STOP_EN if lang == "en" else STOP_FR),
        "punctuation_removed": list(PUNCT),
        "stemmer": "PorterStemmer",
        "lemmatizer": "WordNetLemmatizer"
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

# ============================================================
# PIPELINE D'EXÉCUTION (T1 à T8)
# ============================================================
if __name__ == "__main__":
    filepath = os.path.join("data", "corpus_en.txt")
    
    print("=" * 60)
    print("TÂCHE 1 : CHARGEMENT DU CORPUS")
    print("=" * 60)
    txt = load_text(filepath)
    print(f"Caractères lus : {len(txt)}\nContenu :\n{txt.strip()}\n")

    print("=" * 60)
    print("TÂCHE 2 : SEGMENTATION EN PHRASES (NLTK)")
    print("=" * 60)
    sents = split_sentences(txt)
    for i, s in enumerate(sents, 1):
        print(f"{i:02d}: {s}")

    print("\n" + "=" * 60)
    print("TÂCHE 3 : SEGMENTATION EN MOTS (TOKENS)")
    print("=" * 60)
    toks = tokenize_words(sents)
    for i, sent in enumerate(toks[:2], 1):
        print(f"Phrase {i} : {sent}")

    print("\n" + "=" * 60)
    print("TÂCHE 4 : SUPPRESSION DES STOP WORDS")
    print("=" * 60)
    no_stop = drop_stopwords(toks, lang="en")
    for i, sent in enumerate(no_stop[:2], 1):
        print(f"Phrase {i} : {sent}")

    print("\n" + "=" * 60)
    print("TÂCHE 5 : SUPPRESSION DE LA PONCTUATION")
    print("=" * 60)
    no_punct = drop_punctuation(no_stop)
    for i, sent in enumerate(no_punct[:2], 1):
        print(f"Phrase {i} : {sent}")

    print("\n" + "=" * 60)
    print("TÂCHE 6 : MINUSCULES AVEC EXCEPTIONS (NLP, DATA, USA)")
    print("=" * 60)
    lowered = to_lower_with_exceptions(no_punct)
    for i, sent in enumerate(lowered[:2], 1):
        print(f"Phrase {i} : {sent}")

    print("\n" + "=" * 60)
    print("TÂCHE 7 : RÉDUCTION MORPHOLOGIQUE (STEM vs LEMMA)")
    print("=" * 60)
    stemmed = stem(lowered)
    lemmatized = lemmatize(lowered)
    vocab_stem = sorted(set(flatten(stemmed)))
    vocab_lemma = sorted(set(flatten(lemmatized)))
    print("Vocab (Stemming)     :", vocab_stem)
    print("Vocab (Lemmatisation):", vocab_lemma)
    print(f"Tailles -> Stemming: {len(vocab_stem)} | Lemmatisation: {len(vocab_lemma)}")

    print("\n" + "=" * 60)
    print("TÂCHE 8.1 : COMPARAISON AVEC SPACY")
    print("=" * 60)
    sp_sents, sp_toks = spacy_tokenize_compare(txt)
    print(f"NLTK  -> {len(sents)} phrases | {sum(len(s) for s in toks)} tokens")
    print(f"spaCy -> {len(sp_sents)} phrases | {len(sp_toks)} tokens")

    print("\n" + "=" * 60)
    print("TÂCHE 8.2 : IMPACT DU PRÉTRAITEMENT SUR TF-IDF")
    print("=" * 60)
    clean_docs = [" ".join(sent) for sent in lemmatized]
    raw_top, clean_top = compare_tfidf_impact(sents, clean_docs)
    print("Top termes brut (avec mots vides) :")
    for word, score in raw_top:
        print(f"  {word:<15} (score: {score:.3f})")
    print("Top termes nettoyés (lexique porteur de sens) :")
    for word, score in clean_top:
        print(f"  {word:<15} (score: {score:.3f})")

    print("\n" + "=" * 60)
    print("TÂCHE 8.3 : EXPORT CONFIGURATION")
    print("=" * 60)
    save_config()
    print("Configuration exportée dans preprocess_config.json")