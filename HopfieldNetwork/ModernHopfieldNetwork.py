import numpy as np
from scipy.special import softmax, logsumexp


class ModernHopfieldNetwork:
    def __init__(self, dimension: int, beta: float = 1.0):
        self.dimension = dimension
        self.beta = beta
        self.memories = None

    def store_patterns(self, patterns):
        """
        Store patterns as memory matrix.

        Args:
            patterns: numpy array of shape (dimension, n_patterns)
                     Each column is a stored pattern
        """
        self.memories = patterns.copy()

    def energy(self, query):
        """
        Compute modern Hopfield energy for continuous states.

        Args:
            query: state vector of shape (dimension,)

        Returns:
            scalar energy value
        """
        if self.memories is None:
            raise ValueError("No patterns stored in network")

        # Compute similarities between query and all stored patterns
        similarities = self.beta * (self.memories.T @ query)

        # Modern Hopfield energy: -log(sum(exp(similarities)))
        return -logsumexp(similarities)

    def retrieve_pattern(self, query, mask=None):
        """
        Retrieve pattern using modern Hopfield dynamics with one-step convergence.

        Args:
            query: partial or complete pattern of shape (dimension,)
            mask: boolean array indicating known positions (optional)

        Returns:
            retrieved pattern of shape (dimension,)
        """
        if self.memories is None:
            raise ValueError("No patterns stored in network")

        # Handle masked queries (partial patterns)
        if mask is not None:
            # For unknown positions, use neutral values
            effective_query = query.copy()
            effective_query[~mask] = 0.0
        else:
            effective_query = query.copy()

        # Compute similarities (attention logits)
        similarities = self.beta * (self.memories.T @ effective_query)

        # Apply softmax to get attention weights
        attention_weights = softmax(similarities)

        # Retrieve weighted combination of stored patterns
        retrieved = self.memories @ attention_weights

        # For masked queries, preserve known positions
        if mask is not None:
            retrieved[mask] = query[mask]

        return retrieved

    def complete_pattern(self, partial_query, known_positions):
        """
        Complete partial pattern with known character positions.

        Args:
            partial_query: partial pattern with known positions filled
            known_positions: boolean mask indicating which positions are known

        Returns:
            completed pattern
        """
        return self.retrieve_pattern(partial_query, mask=known_positions)

    def find_best_match(self, query, candidate_words=None):
        """
        Find best matching stored pattern or candidate word.

        Args:
            query: query pattern
            candidate_words: optional list of candidate word encodings

        Returns:
            best matching pattern and its similarity score
        """
        if candidate_words is not None:
            patterns_to_check = np.column_stack(candidate_words)
        else:
            patterns_to_check = self.memories

        similarities = self.beta * (patterns_to_check.T @ query)
        best_idx = np.argmax(similarities)
        best_pattern = patterns_to_check[:, best_idx]
        best_score = similarities[best_idx]

        return best_pattern, best_score


class WordHopfieldNetwork(ModernHopfieldNetwork):
    """
    Specialized Modern Hopfield Network for word completion tasks.
    """

    def __init__(self, word_length: int = 5, alphabet_size: int = 26, beta: float = 2.0):
        dimension = word_length * alphabet_size
        super().__init__(dimension, beta)
        self.word_length = word_length
        self.alphabet_size = alphabet_size

    def encode_word(self, word):
        """
        Convert word to one-hot encoded vector.

        Args:
            word: string of length word_length

        Returns:
            one-hot encoded vector of shape (dimension,)
        """
        if len(word) != self.word_length:
            raise ValueError(
                f"Word must be exactly {self.word_length} characters")

        encoding = np.zeros(self.dimension)
        for pos, char in enumerate(word.lower()):
            if 'a' <= char <= 'z':
                char_idx = ord(char) - ord('a')
                flat_idx = pos * self.alphabet_size + char_idx
                encoding[flat_idx] = 1.0

        return encoding

    def decode_word(self, encoding):
        """
        Convert one-hot encoded vector back to word.

        Args:
            encoding: vector of shape (dimension,)

        Returns:
            decoded word string
        """
        word_chars = []
        for pos in range(self.word_length):
            start_idx = pos * self.alphabet_size
            end_idx = start_idx + self.alphabet_size
            char_probs = encoding[start_idx:end_idx]
            char_idx = np.argmax(char_probs)
            char = chr(char_idx + ord('a'))
            word_chars.append(char)

        return ''.join(word_chars)

    def store_vocabulary(self, words):
        """
        Store list of words as memory patterns.

        Args:
            words: list of strings
        """
        patterns = np.zeros((self.dimension, len(words)))
        for i, word in enumerate(words):
            patterns[:, i] = self.encode_word(word)
        self.store_patterns(patterns)

    def complete_word(self, partial_word, unknown_char='_'):
        """
        Complete partial word using stored vocabulary.

        Args:
            partial_word: string with unknown positions marked by unknown_char
            unknown_char: character representing unknown positions

        Returns:
            completed word string
        """
        # Create known positions mask
        known_positions = np.zeros(self.dimension, dtype=bool)
        partial_encoding = np.zeros(self.dimension)

        for pos, char in enumerate(partial_word):
            if char != unknown_char:
                # Mark this position as known
                start_idx = pos * self.alphabet_size
                end_idx = start_idx + self.alphabet_size
                known_positions[start_idx:end_idx] = True

                # Set the known character
                if 'a' <= char.lower() <= 'z':
                    char_idx = ord(char.lower()) - ord('a')
                    flat_idx = pos * self.alphabet_size + char_idx
                    partial_encoding[flat_idx] = 1.0

        # Retrieve completed pattern
        completed_encoding = self.complete_pattern(
            partial_encoding, known_positions)

        # Apply winner-take-all to ensure valid one-hot encoding
        for pos in range(self.word_length):
            start_idx = pos * self.alphabet_size
            end_idx = start_idx + self.alphabet_size
            position_activations = completed_encoding[start_idx:end_idx]

            # Set winner-take-all: highest activation gets 1, others get 0
            max_idx = np.argmax(position_activations)
            completed_encoding[start_idx:end_idx] = 0.0
            completed_encoding[start_idx + max_idx] = 1.0

        return self.decode_word(completed_encoding)

    def retrieve_possible_words(self, partial_word, unknown_char='_'):
        """
        Find best matching word completion for a partial word.

        Args:
            partial_word: string with unknown positions marked by '_'

        Returns: List[str]
            Matched words
        """
        res = []
        while True:
            completed_word = self.complete_word(partial_word, unknown_char)
            print(f"Completed word: {completed_word}")
            # Check if the retrieved word matches the partial word
            position = np.where(np.all(self.memories == self.encode_word(
                completed_word).reshape(-1, 1), axis=0))
            if position[0].size > 0:
                res.append(completed_word)
                self.memories = np.delete(
                    self.memories, position, axis=1)
            else:
                break

        return res


def check(partial_word, completed_word, unknown_char='_'):
    """
    Check if the completed word matches the partial word.

    Args:
        partial_word: string with unknown positions marked by unknown_char
        completed_word: string to check against the partial word
        unknown_char: character representing unknown positions

    Returns:
        bool: True if completed_word matches partial_word, False otherwise
    """
    if len(partial_word) != len(completed_word):
        return False

    for p_char, c_char in zip(partial_word, completed_word):
        if p_char != unknown_char and p_char != c_char:
            return False

    return True

# Example usage for your specific case


def test_word_completion():
    # Create network for 5-letter words
    network = WordHopfieldNetwork(
        word_length=5, alphabet_size=26, beta=10000.0)

    # Store vocabulary
    vocabulary = ["empty", "sixty", "dabby", "deity", "dusty",
                  "dirty", "party", ]
    network.store_vocabulary(vocabulary)

    # Test completion of "d__ty"
    completed_word = network.complete_word("_____")
    print(f"Completed '_____' as: {completed_word}")

    words = network.retrieve_possible_words("d____")
    print(f"Possible completions for 'd__ty': {words}")

    # # Test with different partial patterns
    # test_cases = ["de___", "____y", "_ei__", "par__"]
    # for partial in test_cases:
    #     completed = network.complete_word(partial)
    #     print(f"Completed '{partial}' as: {completed}")


if __name__ == "__main__":
    test_word_completion()
