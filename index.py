pip install nltk

import nltk
nltk.download("all")
# ------------------------------------------ # PRACTICAL_1 ---------------------------------------------------------------

# PORTER STEMMER

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# create an object of class PorterStemmer
porter = PorterStemmer()
print(porter.stem("play"))
print(porter.stem("playing"))
print(porter.stem("plays"))
print(porter.stem("played"))

sentence = "Programmers program with programming languages"
s2 = "my dog is very playful"
words = word_tokenize(sentence +" "+ s2)

for w in words:
  print(w, " : ", porter.stem(w))

# ------------------------------------------ # PRACTICAL_1 ---------------------------------------------------------------
#Snowball stemming algorithm
from nltk.stem.snowball import SnowballStemmer
snow = SnowballStemmer(language='english')
sentence = "Programmers coded with programming languages and using different framework and technologies"

words = word_tokenize(sentence)

for w in words:
  print(w, " : ", snow.stem(w))


# ------------------------------------------ # PRACTICAL_1 ---------------------------------------------------------------

# Lancaster Stemmer

from nltk.stem import LancasterStemmer
Lanc_stemmer = LancasterStemmer()
sentence = "Programmers program with programming languages"

words = word_tokenize(sentence)

for w in words:
  print(w, " : ", Lanc_stemmer.stem(w))


# ------------------------------------------ # PRACTICAL_1 ---------------------------------------------------------------

from nltk.stem import RegexpStemmer
regexp = RegexpStemmer('ing$|s$|e$|able$|ion$', min=4)
words = ['connecting','connect','connects','fractionally','fractions',"consult","consulation", "consulting", "consults"]
for word in words:
  print(word,"--->",regexp.stem(word))

# ------------------------------------------ # PRACTICAL_1 ---------------------------------------------------------------
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print("rocks :", lemmatizer.lemmatize("rocks"))
print("corpora :", lemmatizer.lemmatize("corpora"))

# a denotes adjective in "pos"
print("better :", lemmatizer.lemmatize("better", pos="a"))

# v denotes verb in "pos"
print("took :", lemmatizer.lemmatize("took", pos="v"))

# ------------------------------------------ # PRACTICAL_1 ---------------------------------------------------------------

import spacy
# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')
# Define a sample text
text = "The quick brown foxes are jumping over the lazy dogs."
# Process the text using spaCy
doc = nlp(text)
# Extract lemmatized tokens
lemmatized_tokens = [token.lemma_ for token in doc]
# Join the lemmatized tokens into a sentence
lemmatized_text = ' '.join(lemmatized_tokens)
# Print the original and lemmatized text
print("Original Text:", text)
print("Lemmatized Text:", lemmatized_text)


# ------------------------------------------ # PRACTICAL_2 ---------------------------------------------------------------
from nltk.tokenize import word_tokenize
text = "GeeksforGeeks is a Computer Science platform."
tokenized_text = word_tokenize(text)
print("Spilt Words: ", tokenized_text)
print("Count of Words: ", len(tokenized_text))

# ------------------------------------------ # PRACTICAL_3 ---------------------------------------------------------------
# FOR ENGLISH
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

def tokenize_and_find_pos(paragraph):
    sentences = sent_tokenize(paragraph)

    for i, sentence in enumerate(sentences, start=1):
        print(f"Sentence {i}: {sentence}")
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        print("Parts of Speech:", pos_tags, "\n")

# Example paragraph
paragraph = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human (natural) languages. It is used to apply algorithms to identify and extract the natural language rules such that the unstructured language data is converted into a form that computers can understand."

tokenize_and_find_pos(paragraph)

# ------------------------------------------ # PRACTICAL_3 ---------------------------------------------------------------
# FOR OTHER
import nltk
nltk.download('punkt')
german_tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
german_tokens=german_tokenizer.tokenize('Wie geht es Ihnen? Gut,danke.')
print(german_tokens)
ps = pos_tag(german_tokens)
print(ps)

# ------------------------------------------ # PRACTICAL_4 ---------------------------------------------------------------

import nltk
from nltk import RegexpParser
from nltk.tokenize import word_tokenize
#from IPython.display import display
# Download necessary NLTK resources
nltk.download('punkt')
# Define a sample grammar rule
grammar = r"""
NP: {<DT|JJ|NN.*>+}  # Chunk sequences of DT, JJ, NN
PP: {<IN><NP>} # Chunk prepositions followed by NP
VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
CLAUSE: {<NP><VP>} # Chunk NP, VP pairs
"""
# Create a chunk parser
chunk_parser = RegexpParser(grammar)
# Define a sample sentence
sentence = "The quick brown fox jumps over the lazy dog"
# Tokenize the sentence
tokens = word_tokenize(sentence)
# Perform POS tagging
tagged_tokens = nltk.pos_tag(tokens)
# Apply chunk parsing
parse_tree = chunk_parser.parse(tagged_tokens)
# Display parse tree
parse_tree.pretty_print()

# ------------------------------------------ # PRACTICAL_5 ---------------------------------------------------------------

import math
from collections import Counter
def calculate_tf(text):
  words = text.lower().split()
  word_count = len(words)
  word_freq = Counter(words)
  tf = {word: freq / word_count for word, freq in word_freq.items()}
  return tf
def calculate_idf(documents):
  total_docs = len(documents)
  idf = {}
  all_words = [word for document in documents for word in
  set(document.lower().split())]
  for word in all_words:
    doc_count = sum([1 for document in documents if word in
    document.lower().split()])
    idf[word] = math.log(total_docs / (1 + doc_count))
  return idf
# Example documents
document1 = "This is the first document. It contains words to analyze term frequency and inverse document frequency."
document2 = "The second document has some overlapping words with the first document but also includes unique terms."
document3 = "Finally, the third document is shorter and has fewer words compared to the other two documents."
documents = [document1, document2, document3]
# Calculate TF for each document
tf_documents = [calculate_tf(document) for document in documents]
# Calculate IDF for all documents
idf = calculate_idf(documents)
print("Term Frequency (TF) for each document:")
for i, tf_doc in enumerate(tf_documents, start=1):
  print(f"Document {i}: {tf_doc}")
  print("\nInverse Document Frequency (IDF) for all words:")
for word, idf_value in idf.items():
  print(f"{word}: {idf_value}")

# ------------------------------------------ # PRACTICAL_6 ---------------------------------------------------------------
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  word_tokens = word_tokenize(text)
  filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
  return ' '.join(filtered_text)
def identify_pos(text):
  sentences = sent_tokenize(text)
  tagged_sentences = [pos_tag(word_tokenize(sentence)) for sentence in sentences]
  return tagged_sentences
# Example paragraph
paragraph = """
Natural Language Processing (NLP) is a subfield of artificial
intelligence concerned with the interaction between computers and`
humans in natural language. It focuses on the interaction between
computers and humans in the natural language and it is a field at the
intersection of computer science, artificial intelligence, and
computational linguistics.
"""
print("paragraph: ", paragraph)
# Remove stop words
paragraph_without_stopwords = remove_stopwords(paragraph)
print("Paragraph without stopwords:")
print(paragraph_without_stopwords)
# Identify Parts of Speech
tagged_sentences = identify_pos(paragraph)
print("\nParts of speech:")
for sentence in tagged_sentences:
  print(sentence)

# ------------------------------------------ # PRACTICAL_7 ---------------------------------------------------------------

# ------------------------------------------ # PRACTICAL_8 ---------------------------------------------------------------
import nltk

from nltk.corpus import wordnet as wn

def calculate_similarities(word1, word2):
  synsets1 = wn.synsets(word1)
  synsets2 = wn.synsets(word2)

  if not synsets1 or not synsets2:
    print(f"No synsets found for one of the words: {word1}, {word2}")
    return

  synset1 = synsets1[0]
  synset2 = synsets2[0]

  path_similarity = synset1.path_similarity(synset2)

  wup_similarity = synset1.wup_similarity(synset2)

  print(f"Path Similarity between '{word1}' and '{word2}':", path_similarity)
  print(f"Wu-Palmer Similarity between '{word1}' and '{word2}':", wup_similarity)

word1 = "consultant”
word2 = "consultancy"

calculate_similarities(word1, word2)

# ------------------------------------------ # PRACTICAL_9 ---------------------------------------------------------------
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.chunk.regexp import RegexpParser
from nltk import ne_chunk

sentence = "In August, India and Microsoft plan to address the issue of climate change and alloted $5000000 for it."

tokens = word_tokenize(sentence)

tagged = pos_tag(tokens)

chunk_pattern = r"NP: {<DT>?<JJ>*<NN>}"
cp = RegexpParser(chunk_pattern)
chunked = cp.parse(tagged)
ne_chunked = ne_chunk(tagged)

print("POS Tags:", tagged)
chunked.pretty_print()

iob_tagged = tree2conlltags(ne_chunked)
print("IOB Tags:", iob_tagged)

entities = ['GPE', 'ORGANIZATION', 'DATE', 'MONEY','COUNTRY']
for token, pos, entity in iob_tagged:
  if entity in entities:
    print(f"{entity}: {token}")

# ------------------------------------------ # PRACTICAL_10 --------------------------------------------------------------
import nltk

from nltk.corpus import wordnet as wn

hello_synsets = wn.synsets('hello')
print("All synsets for 'hello':", hello_synsets)

first_synset = hello_synsets[0]
print("\nFirst Synset:", first_synset)

first_lemma = first_synset.lemmas()[0].name()
print("First lemma name of the 0th Synset:", first_lemma)

synset_name = first_synset.name()
synset_definition = first_synset.definition()
synset_examples = first_synset.examples()
print("\nName of the 0th Synset:", synset_name)
print("Definition of the 0th Synset:", synset_definition)
print("Examples of the 0th Synset:", synset_examples)

synonyms = [lemma.name() for lemma in first_synset.lemmas()]
antonyms = [ant.name() for lemma in first_synset.lemmas() for ant in lemma.antonyms()]
print("\nSynonyms in Synset:", synonyms)
print("Antonyms in Synset:", antonyms)

hypernyms = first_synset.hypernyms()
hyponyms = first_synset.hyponyms()
print("\nHypernyms of the 0th Synset:", hypernyms)
print("Hyponyms of the 0th Synset:", hyponyms)
