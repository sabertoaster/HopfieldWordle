import numpy as np
from collections import Counter
from typing import List, Dict, Any


class LoRA:
    def word_to_vector(self, word):
        array = np.zeros((26, 1))
        for i in range(0, 5):
            array[ord(word[i]) - 97] = 1
        return array

    def LSI(self, possible_words, must_haves):
        A = self.word_to_vector(possible_words[0])
        for word in possible_words:
            A = np.concatenate((A, self.word_to_vector(word)), 1)
        A = np.delete(A, 0, 1)

        AA = np.matmul(A, np.matrix.transpose(A))
        w, v = np.linalg.eigh(AA)

        dom1 = np.abs(v[:, -1])
        angle_vector = np.array([])
        angle_min = 90

        for i in range(0, len(possible_words)):
            angle1 = np.arccos(
                np.matmul(A[:, i], dom1)/(np.linalg.norm(A[:, i])))

            if np.isnan(angle1):
                angle1 = np.pi / 2

            angle_vector = np.append(angle_vector, angle1*180/np.pi)

            if np.abs(angle1) < angle_min:
                angle_min = angle1

        best_words = np.array(possible_words)[angle_vector.argsort()]
        angle_vector = angle_vector[angle_vector.argsort()]
        return best_words, angle_vector


class Solver:
    def __init__(self, word_length=5, init_words=None, hopfield_network=None):
        self.ban_letters = set()
        self.possible_words = init_words
        self.word_length = word_length
        self.hopfield_network = hopfield_network
        self.lora = LoRA()

    def combinations(self, answer: str, hint: str) -> List[str]:
        """
        Generate all possible combinations of letters based on the hint.
        Hint is a string where:
        - '_' means the letter is not in the word
        - '-' means the letter is in the word but in the wrong position
        - 'o' means the letter is in the word and in the correct position
        """
        possible_combinations = [hint.replace('-', '_')]
        idx = 0
        for letter, h in zip(answer, hint):
            if h == '-':
                # Generate combinations with the letter in different positions
                temp_combinations = []
                for el in possible_combinations:
                    for i in range(self.word_length):
                        if el[i] == '_' and idx != i:
                            new_combination = el[:i] + letter + el[i+1:]
                            temp_combinations.append(new_combination)
                possible_combinations = temp_combinations
            idx += 1
        return possible_combinations

    def get_word_from_hint(self, answer: str, hint: str) -> str:
        """
        Answer is the guessed word.
        Hint is the feedback from the game with these rules:
        - '_' means the letter is not in the word
        - '-' means the letter is in the word but in the wrong position
        - 'o' means the letter is in the word and in the correct position
        """
        if self.hopfield_network is None:
            raise ValueError("Hopfield network is not initialized.")

        self.ban_letters.update(
            letter for letter, h in zip(answer, hint) if h == '_')

        hints = self.combinations(answer, hint)
        print(f"Possible combinations based on the hint: {hints}")
        words = set()
        if not hints:
            raise ValueError("No valid combinations found based on the hint.")

        for word in hints:
            words.update(
                self.hopfield_network.retrieve_possible_words(word))
        words = list(words)

        self.possible_words.extend([
            word for word in words
        ])

        # Remove words that contain banned letters
        self.possible_words = [
            word for word in self.possible_words if not any(letter in word for letter in self.ban_letters)
        ]

        word = self.get_best_word()
        self.possible_words.remove(word)
        return word

    def get_best_word(self) -> str:
        """
        Get the best word to guess based on the current possible words.
        The best word is the one that aligns best with eigenvectors of possible words.
        """
        if not self.possible_words:
            raise ValueError("No possible words available.")
        print(f"{self.possible_words}")
        best_words, angles = self.lora.LSI(self.possible_words, [])

        return best_words[0]


def main():
    # Example usage
    solver = Solver(word_length=5, init_words=set(
        ["apple", "grape", "peach", "berry"]))
    answer = "plane"
    hint = "_--_e"
    print(solver.combinations(answer, hint))


if __name__ == "__main__":
    main()
