import numpy as np
from scipy.special import softmax, logsumexp
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class WordPattern:
    """Represents a word pattern with known and unknown positions."""
    pattern: str
    unknown_char: str = '_'

    @property
    def known_positions(self) -> List[int]:
        """Get indices of known character positions."""
        return [i for i, char in enumerate(self.pattern) if char != self.unknown_char]

    @property
    def unknown_positions(self) -> List[int]:
        """Get indices of unknown character positions."""
        return [i for i, char in enumerate(self.pattern) if char == self.unknown_char]


class ModernHopfieldNetwork:
    """
    Modern Hopfield Network implementation with continuous states and one-step convergence.

    This implementation uses exponential energy function and attention-based dynamics
    for pattern retrieval.
    """

    def __init__(self, dimension: int, beta: float = 1.0):
        """
        Initialize Modern Hopfield Network.

        Args:
            dimension: Dimensionality of patterns
            beta: Inverse temperature parameter (higher = sharper attention)
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        if beta <= 0:
            raise ValueError("Beta must be positive")

        self.dimension = dimension
        self.beta = beta
        self.memories = None
        self._n_patterns = 0

    def store_patterns(self, patterns: np.ndarray) -> None:
        """
        Store patterns as memory matrix.

        Args:
            patterns: numpy array of shape (dimension, n_patterns)
                     Each column is a stored pattern
        """
        if patterns.shape[0] != self.dimension:
            raise ValueError(
                f"Pattern dimension {patterns.shape[0]} doesn't match network dimension {self.dimension}")

        self.memories = patterns.copy()
        self._n_patterns = patterns.shape[1]

    @property
    def capacity(self) -> int:
        """Get number of stored patterns."""
        return self._n_patterns

    def energy(self, query: np.ndarray) -> float:
        """
        Compute modern Hopfield energy for continuous states.

        Args:
            query: state vector of shape (dimension,)

        Returns:
            scalar energy value
        """
        self._validate_memory()

        # Compute similarities between query and all stored patterns
        similarities = self.beta * (self.memories.T @ query)

        # Modern Hopfield energy: -log(sum(exp(similarities)))
        return -logsumexp(similarities)

    def retrieve_pattern(self, query: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Retrieve pattern using modern Hopfield dynamics with one-step convergence.

        Args:
            query: partial or complete pattern of shape (dimension,)
            mask: boolean array indicating known positions (optional)

        Returns:
            retrieved pattern of shape (dimension,)
        """
        self._validate_memory()

        # Handle masked queries (partial patterns)
        effective_query = self._prepare_masked_query(query, mask)

        # Compute attention weights
        attention_weights = self._compute_attention_weights(effective_query)

        # Retrieve weighted combination of stored patterns
        retrieved = self.memories @ attention_weights

        # Preserve known positions for masked queries
        if mask is not None:
            retrieved[mask] = query[mask]

        return retrieved

    def find_k_nearest(self, query: np.ndarray, k: int = 1) -> List[Tuple[np.ndarray, float]]:
        """
        Find k nearest stored patterns to query.

        Args:
            query: query pattern
            k: number of nearest patterns to return

        Returns:
            List of (pattern, similarity_score) tuples
        """
        self._validate_memory()

        similarities = self.beta * (self.memories.T @ query)
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(
            similarities[top_k_indices])[::-1]]

        results = []
        for idx in top_k_indices:
            pattern = self.memories[:, idx]
            score = similarities[idx]
            results.append((pattern, score))

        return results

    def _validate_memory(self) -> None:
        """Validate that patterns have been stored."""
        if self.memories is None:
            raise ValueError("No patterns stored in network")

    def _prepare_masked_query(self, query: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Prepare query for masked retrieval."""
        if mask is None:
            return query.copy()

        effective_query = query.copy()
        effective_query[~mask] = 0.0
        return effective_query

    def _compute_attention_weights(self, query: np.ndarray) -> np.ndarray:
        """Compute attention weights for pattern retrieval."""
        similarities = self.beta * (self.memories.T @ query)
        return softmax(similarities)


class WordHopfieldNetwork(ModernHopfieldNetwork):
    """
    Specialized Modern Hopfield Network for word completion tasks.

    This network uses one-hot encoding for characters and provides
    word-specific operations like completion and vocabulary management.
    """

    def __init__(self, word_length: int = 5, alphabet_size: int = 26, beta: float = 2.0):
        """
        Initialize Word Hopfield Network.

        Args:
            word_length: Length of words to store
            alphabet_size: Size of alphabet (default 26 for English)
            beta: Inverse temperature parameter
        """
        if word_length <= 0:
            raise ValueError("Word length must be positive")
        if alphabet_size <= 0:
            raise ValueError("Alphabet size must be positive")

        dimension = word_length * alphabet_size
        super().__init__(dimension, beta)
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self._vocabulary = []

    def encode_word(self, word: str) -> np.ndarray:
        """
        Convert word to one-hot encoded vector.

        Args:
            word: string of length word_length

        Returns:
            one-hot encoded vector of shape (dimension,)
        """
        if len(word) != self.word_length:
            raise ValueError(
                f"Word must be exactly {self.word_length} characters, got {len(word)}")

        encoding = np.zeros(self.dimension)
        for pos, char in enumerate(word.lower()):
            if not 'a' <= char <= 'z':
                raise ValueError(f"Invalid character '{char}' in word")

            char_idx = ord(char) - ord('a')
            if char_idx >= self.alphabet_size:
                raise ValueError(f"Character '{char}' outside alphabet size")

            flat_idx = pos * self.alphabet_size + char_idx
            encoding[flat_idx] = 1.0

        return encoding

    def decode_word(self, encoding: np.ndarray) -> str:
        """
        Convert encoded vector back to word.

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

            if np.sum(char_probs) == 0:
                # Handle empty position
                word_chars.append('?')
            else:
                char_idx = np.argmax(char_probs)
                char = chr(char_idx + ord('a'))
                word_chars.append(char)

        return ''.join(word_chars)

    def store_vocabulary(self, words: List[str]) -> None:
        """
        Store list of words as memory patterns.

        Args:
            words: list of strings
        """
        if not words:
            raise ValueError("Vocabulary cannot be empty")

        # Validate and encode all words
        patterns = np.zeros((self.dimension, len(words)))
        for i, word in enumerate(words):
            patterns[:, i] = self.encode_word(word)

        self.store_patterns(patterns)
        self._vocabulary = words.copy()

    @property
    def vocabulary(self) -> List[str]:
        """Get stored vocabulary."""
        return self._vocabulary.copy()

    def complete_word(self, partial_word: str, unknown_char: str = '_') -> str:
        """
        Complete partial word using stored vocabulary.

        Args:
            partial_word: string with unknown positions marked by unknown_char
            unknown_char: character representing unknown positions

        Returns:
            completed word string
        """
        if len(partial_word) != self.word_length:
            raise ValueError(
                f"Partial word must be {self.word_length} characters")

        # Create encoding and mask for known positions
        partial_encoding, known_mask = self._encode_partial_word(
            partial_word, unknown_char)

        # Retrieve completed pattern
        completed_encoding = self.retrieve_pattern(
            partial_encoding, known_mask)

        # Apply winner-take-all to ensure valid one-hot encoding
        completed_encoding = self._apply_winner_take_all(completed_encoding)

        return self.decode_word(completed_encoding)

    def find_matching_words(self, partial_word: str, unknown_char: str = '_',
                            max_results: int = None) -> List[str]:
        """
        Find all words matching a partial pattern.

        Args:
            partial_word: string with unknown positions marked by unknown_char
            unknown_char: character representing unknown positions
            max_results: maximum number of results to return (None for all)

        Returns:
            List of matching words sorted by similarity
        """
        if len(partial_word) != self.word_length:
            raise ValueError(
                f"Partial word must be {self.word_length} characters")

        # Create encoding for partial word
        partial_encoding, known_mask = self._encode_partial_word(
            partial_word, unknown_char)

        # Find all matching words
        matches = []
        for i, word in enumerate(self._vocabulary):
            word_encoding = self.memories[:, i]

            # Check if word matches the known positions
            if self._matches_pattern(word_encoding, partial_encoding, known_mask):
                # Compute similarity score
                similarity = np.dot(
                    partial_encoding[known_mask], word_encoding[known_mask])
                matches.append((word, similarity))

        # Sort by similarity (all will be 1.0 for exact matches on known positions)
        matches.sort(key=lambda x: x[1], reverse=True)

        # Extract just the words
        result = [word for word, _ in matches]

        if max_results is not None:
            result = result[:max_results]

        return result

    def retrieve_top_k_completions(self, partial_word: str, k: int = 5,
                                   unknown_char: str = '_') -> List[Tuple[str, float]]:
        """
        Retrieve top-k most likely completions for a partial word.

        Args:
            partial_word: string with unknown positions marked
            k: number of completions to return
            unknown_char: character representing unknown positions

        Returns:
            List of (word, score) tuples
        """
        partial_encoding, _ = self._encode_partial_word(
            partial_word, unknown_char)

        # Get k nearest patterns
        nearest_patterns = self.find_k_nearest(
            partial_encoding, k=min(k, self.capacity))

        # Decode patterns to words
        results = []
        for pattern, score in nearest_patterns:
            word = self.decode_word(pattern)
            results.append((word, score))

        return results

    def _encode_partial_word(self, partial_word: str, unknown_char: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode partial word and create mask for known positions.

        Returns:
            Tuple of (partial_encoding, known_mask)
        """
        known_mask = np.zeros(self.dimension, dtype=bool)
        partial_encoding = np.zeros(self.dimension)

        for pos, char in enumerate(partial_word):
            start_idx = pos * self.alphabet_size
            end_idx = start_idx + self.alphabet_size

            if char != unknown_char:
                # Mark this position as known
                known_mask[start_idx:end_idx] = True

                # Set the known character
                if 'a' <= char.lower() <= 'z':
                    char_idx = ord(char.lower()) - ord('a')
                    flat_idx = pos * self.alphabet_size + char_idx
                    partial_encoding[flat_idx] = 1.0

        return partial_encoding, known_mask

    def _apply_winner_take_all(self, encoding: np.ndarray) -> np.ndarray:
        """Apply winner-take-all to ensure valid one-hot encoding."""
        result = np.zeros_like(encoding)

        for pos in range(self.word_length):
            start_idx = pos * self.alphabet_size
            end_idx = start_idx + self.alphabet_size
            position_activations = encoding[start_idx:end_idx]

            # Set winner-take-all: highest activation gets 1, others get 0
            if np.any(position_activations > 0):
                max_idx = np.argmax(position_activations)
                result[start_idx + max_idx] = 1.0

        return result

    def _matches_pattern(self, word_encoding: np.ndarray, partial_encoding: np.ndarray,
                         known_mask: np.ndarray) -> bool:
        """Check if word encoding matches the partial pattern at known positions."""
        return np.allclose(word_encoding[known_mask], partial_encoding[known_mask])


def demo_word_completion():
    """Demonstrate word completion capabilities."""
    print("=== Modern Hopfield Network Word Completion Demo ===\n")

    # Create network for 5-letter words
    network = WordHopfieldNetwork(word_length=5, alphabet_size=26, beta=10.0)

    # Store vocabulary
    vocabulary = [
        "empty", "sixty", "dabby", "deity", "dusty",
        "dirty", "party", "delta", "debug", "depth"
    ]
    network.store_vocabulary(vocabulary)
    print(f"Stored vocabulary: {vocabulary}\n")

    # Test various completion scenarios
    test_cases = [
        ("d__ty", "Words matching 'd__ty'"),
        ("de___", "Words starting with 'de'"),
        ("____y", "Words ending with 'y'"),
        ("_a___", "Words with 'a' as second letter"),
        ("d____", "Words starting with 'd'"),
    ]

    for partial, description in test_cases:
        print(f"{description} (pattern: '{partial}'):")

        # Method 1: Complete using network dynamics
        completed = network.complete_word(partial)
        print(f"  Network completion: {completed}")

        # Method 2: Find all matches
        matches = network.find_matching_words(partial)
        print(f"  All matches: {matches}")

        # Method 3: Get top-k completions with scores
        top_completions = network.retrieve_top_k_completions(partial, k=3)
        print(
            f"  Top completions: {[f'{word} (score: {score:.2f})' for word, score in top_completions]}")
        print()


if __name__ == "__main__":
    demo_word_completion()
