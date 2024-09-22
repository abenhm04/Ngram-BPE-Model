import argparse
import pickle
import random
import re
from collections import defaultdict, Counter, OrderedDict

file_path = "MobyDick.txt" #or whatever .txt file you would like to use.


def read_file(file_path):
    """
    Reads file and returns list of words.
    :param file_path:
    :return:
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data


class NGramModel:

    def __init__(self, n):
        """
        Initializes the NGramModel.
        :param n:
        """
        self.n = n
        self.vocab = set()
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()    #How often a word occurs

    def tokenize(self, text):
        """
        Tokenizes the text and adds it to the vocabulary.
        :param text:
        :return:
        """
        return re.findall(r'\b\w+\b|[^\w\s]', text.lower())

    def train(self, data):
        """
        Trains the NGramModel and stores it in the vocabulary.
        :param data:
        :return:
        """
        tokens = self.tokenize(data)
        self.vocab.update(tokens)

        if self.n == 2:  # Bigram model training logic
            for i in range(len(tokens) - 1):
                word = tokens[i]
                next_word = tokens[i + 1]
                self.ngram_counts[word][next_word] += 1
                self.context_counts[word] += 1

        elif self.n == 1:  # Unigram model training logic
            for word in tokens:
                self.ngram_counts[word][""] += 1
                self.context_counts[word] += 1

    def predict_next_word(self, input: tuple, deterministic: bool = False,
                          previous_word=None):
        """
        Predicts the next word of the input sentence using the NGramModel.
        :param input:
        :param deterministic:
        :param previous_word:
        :return:
        """
        if self.n == 1:  # Unigram model
            if deterministic:
                candidates = {word: count for word, count in
                              self.context_counts.items()
                              if word != previous_word}
                if not candidates:
                    return max(self.context_counts, key=self.context_counts.get)
                return max(candidates, key=candidates.get)
            else:
                words = list(self.context_counts.keys())
                probabilities = [self.context_counts[word] for word in words]
                return random.choices(words, probabilities)[0]

        elif self.n == 2 and len(input) == 2:  # Bigram model
            word1 = input[-1]  # Only consider the last word for bigram prediction
            if word1 not in self.ngram_counts:
                return None

            next_word_probability = self.ngram_counts[word1]

            if deterministic:
                candidates = {word: count for word, count in
                              next_word_probability.items() if word != previous_word}
                if not candidates:
                    return max(next_word_probability, key=next_word_probability.get)
                return max(candidates, key=candidates.get)
            else:
                words = list(next_word_probability.keys())
                probabilities = list(next_word_probability.values())
                return random.choices(words, probabilities)[0]


class BPE:
    """
    BPE class.
    """

    def __init__(self):
        """
        Initializes BPE class.
        """
        self.vocabulary = []   # Stores the tokens and subwords
        self.token_map = {}    # Maps tokens to unique token IDs
        self.token_count = 1   # Counter to assign unique token IDs

    def train(self, data, k=500):
        """
        Trains BPE.
        :param data:
        :param k:
        :return:
        """
        corpus = list(data)  # Turn the input string into a list of characters
        seen = OrderedDict()

        # Populate vocabulary with unique characters
        for char in corpus:
            if char not in seen:
                seen[char] = None

        # Initialize the vocabulary with unique characters
        self.vocabulary = list(seen.keys())
        print(f"Initial vocabulary: {self.vocabulary}") #print statement for debugging not necessary for final prog

        # Assign token IDs to each character in the vocabulary
        for token in self.vocabulary:
            self.token_map[token] = self.token_count
            print(f"Assigning token ID {self.token_count} to character '{token}'")
            self.token_count += 1

        # Merge the most frequent pairs for k iterations
        for iteration in range(k):
            pairs = Counter()

            # Count frequency of adjacent pairs
            for j in range(len(corpus) - 1):
                pair = corpus[j] + corpus[j + 1]
                pairs[pair] += 1

            # Stop if no pairs exist
            if not pairs:
                break

            # Find the most frequent pair
            most_frequent = pairs.most_common(1)[0][0]
            print(f"Iteration {iteration + 1}: Most frequent pair is '{most_frequent}'")

            # Add the merged pair to the vocabulary
            if most_frequent not in self.vocabulary:
                self.vocabulary.append(most_frequent)
                self.token_map[most_frequent] = self.token_count
                print(f"Assigning token ID {self.token_count} to merged token '{most_frequent}'")
                self.token_count += 1

            # Replace instances of the most frequent pair in the corpus
            i = 0
            while i < len(corpus) - 1:
                if corpus[i] + corpus[i + 1] == most_frequent:
                    corpus[i:i + 2] = [most_frequent]  # Merge the pair
                else:
                    i += 1

        return self.vocabulary

    def tokenize(self, input):
        """
        Tokenize the input string.
        :param input:
        :return:
        """
        char_string = list(input)  # Convert the input string to a list of characters
        tokens = []
        token_ids = []

        # Apply the BPE tokenization to the input string based on the vocabulary
        for token in self.vocabulary:
            if len(token) > 1:  # Only check pairs or larger subwords
                i = 0
                while i < len(char_string) - 1:
                    if char_string[i] + char_string[i + 1] == token:
                        char_string[i:i + 2] = [token]  # Merge the pair
                    else:
                        i += 1

        # Generate tokens and token IDs
        for char in char_string:
            tokens.append(char)
            token_ids.append(self.token_map[char])

        return tokens, token_ids


def main():
    """
    Main function that essentially just sets up all the necessary command line arguments
    :return:
    """
    parser = argparse.ArgumentParser(description='N-gram Model and BPE Tokenizer')
    parser.add_argument('activity', choices=['train_ngram', 'predict_ngram',
                                             'train_bpe', 'tokenize'],
                        help='Activity to perform')
    parser.add_argument('--data', help='Path to the training data corpus')
    parser.add_argument('--save', help='Path to save the trained model')
    parser.add_argument('--load', help='Path to load the trained model')
    parser.add_argument('--word', help='First word(s) for prediction')
    parser.add_argument('--nwords', type=int, help='Number of words to predict')
    parser.add_argument('--text', help='Text to tokenize')
    parser.add_argument('--n', type=int, choices=[1, 2], help='Order of the n-gram model')
    parser.add_argument('--d', action='store_true', help='Deterministic prediction flag')

    args = parser.parse_args()

    if args.activity == 'train_ngram':
        data = read_file(args.data)
        model = NGramModel(args.n)
        model.train(data)
        with open(args.save, 'wb') as f:
            pickle.dump(model, f)


    elif args.activity == 'predict_ngram':

        with open(args.load, 'rb') as f:

            model = pickle.load(f)

        input_words = tuple(args.word.split())

        output = list(input_words)

        #Makes sure we have only two words in input_words for the bigram model

        if model.n == 2 and len(input_words) > 2:
            input_words = input_words[-2:]

        for _ in range(args.nwords):

            next_word = model.predict_next_word(input_words, deterministic=args.d)

            if next_word is None:
                break

            output.append(next_word)

            if model.n == 1:

                input_words = (next_word,)  #Updates unigram

            elif model.n == 2:

                input_words = (input_words[-1], next_word)  # Bigram: update with the last two words
        print(' '.join(output))


    elif args.activity == 'train_bpe':
        data = read_file(args.data)
        bpe_model = BPE()
        bpe_model.train(data, k=500) #Edit to make 3000 for question 4
        with open(args.save, 'wb') as f:
            pickle.dump(bpe_model, f)

    elif args.activity == 'tokenize':
        with open(args.load, 'rb') as f:
            bpe_model = pickle.load(f)
        tokens, token_ids = bpe_model.tokenize(args.text)
        print('Tokens:', tokens)
        print('Token IDs:', token_ids)


if __name__ == '__main__':
    main()

