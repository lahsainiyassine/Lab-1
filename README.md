Description des tâches
Tâche 1 : Lecture du corpus
Fonction : load_text(filepath)

Chargement du fichier source en mémoire avec gestion de l'encodage UTF-8.

Tâche 2 : Segmentation en phrases
Fonction : split_sentences(text)

Découpage du texte en phrases indépendantes à l'aide de sent_tokenize(NLTK), capable de distinguer les points finaux des abréviations usuelles.

Tâche 3 : Segmentation en mots (Tokenisation)
Fonction : tokenize_words(sentences)

Séparation des unités lexicales et de la ponctuation via word_tokenize. Le résultat conserve la hiérarchie initiale sous forme d'une liste de listes ( list[list[str]]), où chaque sous-liste représente une phrase.

Tâche 4 : Filtrage des mots vides
Fonction : drop_stopwords(tokenized, lang="en")

Élimination des termes très fréquents et peu informatifs (articles, prépositions, auxiliaires) à partir du corpus stopwordsde NLTK. La comparaison est effectuée en minuscules pour éviter les taux liés à la casse.

Tâche 5 : Élimination de la ponctuation
Fonction : drop_punctuation(tokenized)

Suppression de tous les caractères de ponctuation standard ( string.punctuation) afin de ne retenir que le texte exploitable.

Tâche 6 : Mise en minuscules et gestion d'exceptions
Fonction : to_lower_with_exceptions(tokenized)

Conversion générale en minuscules pour uniformiser le vocabulaire, tout en conservant intacts les acronymes spécifiés dans EXCEPT_UPPER(par exemple : NLP, DATA, USA).

Tâche 7 : Réduction morphologique (Stemming vs Lemmatisation)
Fonctions : stem(tokenized) etlemmatize(tokenized)

Racinisation (PorterStemmer) : Coupe les affixes de manière algorithmique. Le processus est rapide et réduit fortement le lexique, mais génère des radicaux tronqués hors vocabulaire (ex. fascin , machin ).

Lemmatisation (WordNetLemmatizer) : Recherche la forme canonique dans un dictionnaire. Les mots restent lisibles et corrects (ex.words devient word ) , mais le lemmatiseur par défaut traite les mots comme des noms sans analyse morphosyntaxique plus fine.

Tâche 8 : Extensions
8.1 — Comparaison spaCy vs NLTK : Exécution du modèle neuronal en_core_web_smde spaCy et comparaison du nombre de phrases et de tokens détectés par rapport aux expressions régulières de NLTK.

8.2 — Impact sur TF-IDF : Vectorisation comparée avec TfidfVectorizerde scikit-learn avant et après nettoyage pour mesurer le filtrage pour éliminer le bruit au profit des termes porteurs de sens.

8.3 — Journalisation des paramètres : Export des hyperparamètres du pipeline (langue, règles de filtrage, modèles utilisés) au format JSON dans preprocess_config.json.