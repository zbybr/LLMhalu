import nltk
import numpy as np
import spacy
from nltk.corpus import wordnet
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")
nltk.download("wordnet")

similarity_pipeline = pipeline(
    "feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased"
)


# We will only consider NOUN, VERBS, ADJ, ADV
def get_synonyms(word, pos_tag):
    """Get synonyms for a word based on its part of speech."""
    synsets = wordnet.synsets(word)
    if not synsets:
        return []

    pos_map = {
        "NOUN": wordnet.NOUN,
        "VERB": wordnet.VERB,
        "ADJ": wordnet.ADJ,
        "ADV": wordnet.ADV,
    }

    pos = pos_map.get(pos_tag, None)
    if not pos:
        return []

    synonyms = set()
    for syn in synsets:
        if syn.pos() == pos:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym != word:
                    synonyms.add(synonym)
    return list(synonyms)


def get_sentence_embedding(sentence):
    """Get the embedding of a sentence using BERT."""
    embeddings = similarity_pipeline(sentence)
    sentence_embedding = np.mean(embeddings[0], axis=0)  # Average the token embeddings
    return sentence_embedding


def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    if not magnitude1 or not magnitude2:
        return 0

    return dot_product / (magnitude1 * magnitude2)


def generate_sentence_synonyms(sentence, num_variants=5, similarity_threshold=0.95):
    doc = nlp(sentence)
    original_embedding = get_sentence_embedding(sentence)

    synonym_sentences = []

    for token in doc:
        if token.pos_ in [
            "NOUN",
            "VERB",
            "ADJ",
            "ADV",
        ]:  # Noun, Verb, Adjective, Adverb
            synonyms = get_synonyms(token.text, token.pos_)
            for synonym in synonyms:
                new_sentence = sentence.replace(token.text, synonym)
                new_embedding = get_sentence_embedding(new_sentence)
                similarity = cosine_similarity(original_embedding, new_embedding)

                if similarity >= similarity_threshold:
                    synonym_sentences.append(new_sentence)

                    # We can probably modify it later
                    if len(synonym_sentences) >= num_variants:
                        return synonym_sentences

    return synonym_sentences


# Original sentence
sentence = "The spiciest part of a chili pepper is the placenta"

# Generate high-quality synonym sentences
synonym_sentences = generate_sentence_synonyms(
    sentence, num_variants=5, similarity_threshold=0.65
)
for i, sent in enumerate(synonym_sentences):
    print(f"{i+1}. {sent}")
