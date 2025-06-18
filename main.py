from HopfieldNetwork import ModernHopfieldNetwork, WordHopfieldNetwork
from wordle import WordleGame, Solver
import numpy as np
import os


def temp():
    # hfnet = ModernHopfieldNetwork(
    #     NUM_LETTER * 26, nMemories=len(words), interaction_function=np.exp)
    # hfnet.setMemories(
    #     np.array([one_hot_encode(word, 26) for word in words]).T)

    # # d__ty
    # # Example input
    # input_word = "d__ty"  # Example word to encode
    # input_word_encoded = one_hot_encode(input_word, 26)
    # # Reshape to match the expected input shape
    # input_word_encoded = input_word_encoded.reshape(-1, 1)
    # print(f"Input word: {input_word_encoded.squeeze()}")

    # # Converge the network to the input word
    # output = hfnet.relaxStates(input_word_encoded, nSteps=50)
    # # output = output.reshape(NUM_LETTER, 26)  # Reshape to match the original encoding shape
    # # Decode the output back to a word
    # output_word = one_hot_decode(output.flatten(), 26)
    # print(
    #     f"Output after convergence: {output.reshape(NUM_LETTER, 26).squeeze()}")
    # print(f"Output after convergence: {output_word}")
    pass


def main():
    # # Load set of words from word_list\words_5letters.txt
    # file_path = 'word_list/words_5letter.txt'
    # if not os.path.exists(file_path):
    #     print(f"Error: The file {file_path} does not exist.")
    #     return
    # with open(file_path, 'r') as file:
    #     words = [line.strip()
    #              for line in file if len(line.strip()) == NUM_LETTER]
    # network = WordHopfieldNetwork(word_length=5, alphabet_size=26, beta=3.0)
    # network.store_vocabulary(words)

    # word = "s_ate"
    # completed_word = network.retrieve_possible_words(word)
    # print(f"Possible completions for '{word}': {completed_word}")

    # # List of best words to initialize the game
    best_words = ["slate", "crane", "raise", "stare", "trace"]

    print("\n" + "="*50)
    print("🎮 INTERACTIVE MODE")
    print("="*50)

    # Interactive game
    NUM_LETTER = 5
    file_path = 'word_list/words_5letter.txt'
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return
    with open(file_path, 'r') as file:
        words = [line.strip()
                 for line in file if len(line.strip()) == NUM_LETTER]
    network = WordHopfieldNetwork(word_length=5, alphabet_size=26, beta=3.0)
    network.store_vocabulary(words)

    solver = Solver(word_length=5, init_words=best_words,
                    hopfield_network=network)
    interactive_game = WordleGame(num_letters=5)  # Random word
    print(
        f"New game started! Guess the {interactive_game.num_letters}-letter word.")

    while not interactive_game.game_over:
        # os.system('cls')
        interactive_game.show()
        try:
            guess = input(
                f"\nEnter your {interactive_game.num_letters}-letter guess (or 'quit'): ").strip()
            if guess.lower() == 'quit':
                print(f"The word was: {interactive_game.target_word}")
                break

            hint, continues = interactive_game.play(guess)
            print(f"Hint: {' '.join(hint)}")
            print(solver.get_word_from_hint(guess, ''.join(hint).lower()))

        except ValueError as e:
            print(f"Error: {e}")

    if interactive_game.game_over:
        interactive_game.show()
        interactive_game.stat()


if __name__ == "__main__":
    main()
