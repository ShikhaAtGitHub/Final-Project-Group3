import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
from similarity.normalized_levenshtein import NormalizedLevenshtein
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
import numpy as np
import argparse

# trained_model_path = '/Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/model/'
# trained_tokenizer = '/Users/shikharaikhare/Documents/Course_work/NLP/NLP_Final_Project/Final-Project-Group3/Code/tokenizer/'

s2v = Sense2Vec().from_disk('s2v_old')
sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')
normalized_levenshtein = NormalizedLevenshtein()
tokenizer = None
device = None
model = None

def filter_same_sense_words(original, wordlist):
    filtered_words = []
    base_sense = original.split('|')[1]
    # print(base_sense)
    for eachword in wordlist:
        if eachword[0].split('|')[1] == base_sense:
            filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
    return filtered_words

def get_highest_similarity_score(wordlist, wrd):
    score = []
    for each in wordlist:
        score.append(normalized_levenshtein.similarity(each.lower(), wrd.lower()))
    return max(score)

def sense2vec_get_words(word, s2v, topn, question):
    output = []
    print("word ", word)
    try:
        sense = s2v.get_best_sense(word, senses = ["NOUN", "PERSON", "PRODUCT", "LOC", "ORG", "EVENT", "NORP", "WOKK OF ART", "FAC", "GPE", "NUM", "FACILITY"])
        most_similar = s2v.most_similar(sense, n=topn)
        output = filter_same_sense_words(sense, most_similar)
        # print("Similar ", output)
    except:
        output = []

    threshold = 0.6
    final = [word]
    checklist = question.split()
    for x in output:
        if get_highest_similarity_score(final, x) < threshold and x not in final and x not in checklist:
            final.append(x)

    return final[1:]

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def get_distractors_wordnet(word):
    distractors = []
    try:
        syn = wn.synset(word, 'n')[0]

        word = word.lower()
        orig_word = word
        if len(word.split()) > 0:
            word = word.replace(" ", "_")
        hypernym = syn.hypernyms()
        if len(hypernym) == 0:
            return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name()
            if name == orig_word:
                continue
            name = name.replace("_", " ")
            name = " ".join(w.capitalize() for w in name.split())
            if name is not None and name not in distractors:
                distractors.append(name)

    except:
        print("Wordnet distractors not found")
    return distractors

def get_distractors(word, origsentence, sense2vecmodel, sentencemodel, top_n, lambdaval):
    distractors = sense2vec_get_words(word, sense2vecmodel, top_n, origsentence)
    # print("distractors: ", distractors)
    if len(distractors) == 0:
        return distractors
    distractors_new = [word.capitalize()]
    distractors_new.extend(distractors)

    embedding_sentence = origsentence + " " + word.capitalize()

    keyword_embedding = sentencemodel.encode([embedding_sentence])
    distractor_embeddings = sentencemodel.encode(distractors_new)

    max_keywords = min(len(distractors_new), 4)
    filtered_keywords = mmr(keyword_embedding, distractor_embeddings, distractors_new, max_keywords, lambdaval)

    final = [word.capitalize()]
    for wrd in filtered_keywords:
        if wrd.lower() != word.lower():
            final.append(wrd.capitalize())
    final = final[1:]
    return final

def process_input(context, answer):
    # Construct the input for the model
    text = f"context: {context} answer: {answer} </s>"
    encoding = tokenizer.encode_plus(text, max_length=512, padding=True, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    # Generate output using the model
    model.eval()
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_length=72,
        early_stopping=True,
        num_beams=5,
        num_return_sequences=3
    )

    # Process the outputs to get the distractors or other results
    results = []
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        distractors = get_distractors(answer, sent, s2v, sentence_transformer_model, 40, 0.2)
        results.append({'sentence': sent, 'distractors': distractors})

    return results


def main():
    parser = argparse.ArgumentParser(description='Train and save model and tokenizer paths')
    parser.add_argument('--model_path', type=str, help='Path to save the model')
    parser.add_argument('--tokenizer_path', type=str, help='Path to save the tokenizer')

    args = parser.parse_args()

    if args.model_path is None or args.tokenizer_path is None:
        print("Please provide paths to the model and tokenizer using --model_path and --tokenizer_path arguments.")
        return

    trained_model_path = args.model_path
    trained_tokenizer = args.tokenizer_path

    model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
    tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("device ",device)
    model = model.to(device)

    # context ="In Tokyo Olympics, the USA team defeated France 87-82 to win the gold medal"
    # answer = "USA team"
    # text = "context: "+context + " " + "answer: " + answer + " </s>"
    # print (text)

if __name__ == '__main__':
    main()