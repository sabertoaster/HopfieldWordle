import random
import os
from typing import List, Optional, Tuple

class WordleGame:
    """
    A complete Wordle game implementation with extensible word lengths.
    
    The game state can be thought of as an optimization problem where each guess
    provides constraints that narrow the solution space - similar to how neural
    networks converge on solutions through iterative feedback.
    """
    
    def __init__(self, word: Optional[str] = None, num_letters: int = 5):
        """
        Initialize a new Wordle game.
        
        Args:
            word: Target word (if None, randomize from word list)
            num_letters: Length of words to use (default 5)
        """
        self.num_letters = num_letters
        self.max_attempts = 6
        self.attempts = 0
        self.game_over = False
        self.won = False
        
        # Game history: list of (guess, hint) tuples
        self.history: List[Tuple[str, List[str]]] = []
        
        # Set target word
        if word is None:
            self.target_word = self._load_random_word()
        else:
            if len(word) != num_letters:
                raise ValueError(f"Word must be {num_letters} letters long")
            self.target_word = word.upper()
        
        # Track letter states across all guesses
        self.letter_states = {}  # 'A': 'green'/'yellow'/'gray'
        
        # Load valid word list for validation
        self.valid_words = self._load_word_list()
        
    def _load_random_word(self) -> str:
        """
        Load a random word from word list file.
        Creates a sample word list if file doesn't exist.
        """
        filename = f".\\word_list\\words_{self.num_letters}letter.txt"
        
        # Create sample word list if file doesn't exist
        if not os.path.exists(filename):
            self._create_sample_wordlist(filename)
        
        try:
            with open(filename, 'r') as f:
                words = [line.strip().upper() for line in f if len(line.strip()) == self.num_letters]
            
            if not words:
                raise ValueError(f"No {self.num_letters}-letter words found in {filename}")
                
            return random.choice(words)
            
        except FileNotFoundError:
            # Fallback words
            fallback_words = {
                5: ["APPLE", "HOUSE", "PEACE", "LIGHT", "BRAIN", "SMART", "LEARN", "THINK"],
                4: ["MIND", "CODE", "DATA", "TECH"],
                6: ["NEURAL", "MATRIX", "COGNITIVE"]
            }
            
            if self.num_letters in fallback_words:
                return random.choice(fallback_words[self.num_letters])
            else:
                return "A" * self.num_letters  # Ultimate fallback
    
    def _create_sample_wordlist(self, filename: str):
        """Create a sample word list file for testing."""
        sample_words = {
            5: ["APPLE", "HOUSE", "PEACE", "LIGHT", "BRAIN", "SMART", "LEARN", "THINK", 
                "SPACE", "MOUSE", "CHAIR", "PLANT", "WATER", "EARTH", "STORM", "QUIET"],
            4: ["MIND", "CODE", "DATA", "TECH", "BYTE", "CHIP", "WIRE", "NODE"],
            6: ["NEURAL", "MATRIX", "SIMPLE", "ROBUST", "SYSTEM", "RANDOM"]
        }
        
        words = sample_words.get(self.num_letters, [])
        if words:
            with open(filename, 'w') as f:
                for word in words:
                    f.write(word + '\n')
    
    def _load_word_list(self) -> set:
        """
        Load the complete valid word list for word validation.
        
        Returns:
            Set of valid words in uppercase
        """
        filename = f".\\word_list\\words_{self.num_letters}letter.txt"
        
        # Ensure word list file exists
        if not os.path.exists(filename):
            self._create_sample_wordlist(filename)
        
        try:
            with open(filename, 'r') as f:
                words = {line.strip().upper() for line in f 
                        if len(line.strip()) == self.num_letters and line.strip().isalpha()}
            return words
            
        except FileNotFoundError:
            # Fallback to sample words
            fallback_words = {
                5: {"APPLE", "HOUSE", "PEACE", "LIGHT", "BRAIN", "SMART", "LEARN", "THINK", 
                    "SPACE", "MOUSE", "CHAIR", "PLANT", "WATER", "EARTH", "STORM", "QUIET"},
                4: {"MIND", "CODE", "DATA", "TECH", "BYTE", "CHIP", "WIRE", "NODE"},
                6: {"NEURAL", "MATRIX", "SIMPLE", "ROBUST", "SYSTEM", "RANDOM"}
            }
            return fallback_words.get(self.num_letters, {self.target_word})
    
    def is_valid_word(self, word: str) -> bool:
        """
        Check if a word is in the valid word list.
        
        Args:
            word: Word to validate
            
        Returns:
            True if word is valid, False otherwise
        """
        if not word or len(word) != self.num_letters:
            return False
            
        return word.upper() in self.valid_words
    
    def get_word_list_info(self) -> dict:
        """
        Get information about the loaded word list.
        
        Returns:
            Dictionary with word list statistics
        """
        return {
            'total_words': len(self.valid_words),
            'word_length': self.num_letters,
            'target_in_list': self.target_word in self.valid_words,
            'sample_words': list(self.valid_words)[:10] if self.valid_words else []
        }
    
    def _generate_hint(self, target: str, guess: str) -> List[str]:
        """
        Generate Wordle-style hints using the algorithm from previous implementation.
        
        Returns:
            List of hints where each element is either:
            - The letter itself (green - correct position)
            - '-' (yellow - wrong position)
            - '_' (gray - not in word or used up)
        """
        result = ['_'] * self.num_letters
        target_letters = list(target)
        
        # First pass: Mark exact matches (Green)
        for i in range(self.num_letters):
            if guess[i] == target[i]:
                result[i] = guess[i]  # Green
                target_letters[i] = None  # Remove from available pool
        
        # Second pass: Mark wrong positions (Yellow) or not found (Gray)
        for i in range(self.num_letters):
            if result[i] == '_':  # Not already marked green
                if guess[i] in target_letters:
                    result[i] = '-'  # Yellow
                    target_letters[target_letters.index(guess[i])] = None
        
        return result
    
    def _update_letter_states(self, guess: str, hint: List[str]):
        """Update the overall letter state tracking."""
        for i, letter in enumerate(guess):
            if hint[i] == letter:  # Green
                self.letter_states[letter] = 'green'
            elif hint[i] == '-':  # Yellow
                if self.letter_states.get(letter) != 'green':
                    self.letter_states[letter] = 'yellow'
            else:  # Gray
                if letter not in self.letter_states:
                    self.letter_states[letter] = 'gray'
    
    def play(self, word: str) -> Tuple[List[str], bool]:
        """
        Play a turn with the given word.
        
        Args:
            word: The guessed word
            
        Returns:
            Tuple of (hint, game_continues)
            
        Raises:
            ValueError: If word length is wrong, game is over, or word is not valid
        """
        if self.game_over:
            raise ValueError("Game is already over!")
        
        word = word.upper()
        if len(word) != self.num_letters:
            raise ValueError(f"Word must be {self.num_letters} letters long")
        
        # Validate word is in word list
        if not self.is_valid_word(word):
            raise ValueError(f"'{word}' is not a valid {self.num_letters}-letter word in our dictionary")
        
        # Generate hint
        hint = self._generate_hint(self.target_word, word)
        
        # Update game state
        self.history.append((word, hint))
        self.attempts += 1
        self._update_letter_states(word, hint)
        
        # Check win condition
        if word == self.target_word:
            self.won = True
            self.game_over = True
            
        # Check lose condition
        elif self.attempts >= self.max_attempts:
            self.game_over = True
            
        return hint, not self.game_over
    
    def show(self):
        """Display the current game state."""
        print(f"\n{'='*50}")
        print(f"WORDLE - {self.num_letters} Letter Game")
        print(f"{'='*50}")
        print(f"Attempts: {self.attempts}/{self.max_attempts}")
        
        if self.game_over:
            if self.won:
                print(f"🎉 CONGRATULATIONS! You won in {self.attempts} attempts!")
            else:
                print(f"💀 Game Over! The word was: {self.target_word}")
        
        print(f"\nTarget: {'?' * self.num_letters}")
        print(f"{'='*50}")
        
        # Show guess history
        for i, (guess, hint) in enumerate(self.history):
            print(f"Guess {i+1}: {' '.join(guess)}")
            print(f"Hint  {i+1}: {' '.join(hint)}")
            print("-" * 30)
        
        # Show remaining attempts
        remaining = self.max_attempts - self.attempts
        for i in range(remaining):
            print(f"Guess {self.attempts + i + 1}: {' '.join(['?'] * self.num_letters)}")
            print(f"Hint  {self.attempts + i + 1}: {' '.join(['?'] * self.num_letters)}")
            print("-" * 30)
        
        # Show letter states
        print(f"\nLetter Status:")
        print("🟩 = Correct position, 🟨 = Wrong position, ⬜ = Not in word")
        
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for letter in alphabet:
            if letter in self.letter_states:
                state = self.letter_states[letter]
                if state == 'green':
                    print(f"🟩{letter}", end=" ")
                elif state == 'yellow':
                    print(f"🟨{letter}", end=" ")
                else:
                    print(f"⬜{letter}", end=" ")
            else:
                print(f"⚪{letter}", end=" ")
        print("\n")
    
    def stat(self):
        """Display game statistics (placeholder for future implementation)."""
        print(f"\n📊 GAME STATISTICS")
        print(f"Target word length: {self.num_letters}")
        print(f"Current attempts: {self.attempts}/{self.max_attempts}")
        print(f"Game status: {'Won' if self.won else 'Lost' if self.game_over else 'In Progress'}")
        print(f"Efficiency: {(self.attempts/self.max_attempts)*100:.1f}% of max attempts used")
        
        # Word list info
        word_info = self.get_word_list_info()
        print(f"Dictionary size: {word_info['total_words']} valid {self.num_letters}-letter words")
        
        if self.history:
            print(f"Letters discovered: {len(self.letter_states)} unique letters tested")
            green_letters = sum(1 for state in self.letter_states.values() if state == 'green')
            yellow_letters = sum(1 for state in self.letter_states.values() if state == 'yellow')
            print(f"Progress: {green_letters} correct positions, {yellow_letters} correct letters")
    
    def reset(self, new_word: Optional[str] = None):
        """Reset the game with a new word."""
        self.__init__(new_word, self.num_letters)
    
    def get_valid_letters(self) -> dict:
        """Return sets of letters by their status for AI/solver assistance."""
        return {
            'green': {k for k, v in self.letter_states.items() if v == 'green'},
            'yellow': {k for k, v in self.letter_states.items() if v == 'yellow'},
            'gray': {k for k, v in self.letter_states.items() if v == 'gray'}
        }


# Demo usage and testing
if __name__ == "__main__":
    
    print("\n" + "="*50)
    print("🎮 INTERACTIVE MODE")
    print("="*50)
    
    # Interactive game
    interactive_game = WordleGame()  # Random word
    print(f"New game started! Guess the {interactive_game.num_letters}-letter word.")
    
    while not interactive_game.game_over:
        os.system('cls') 
        interactive_game.show()
        try:
            guess = input(f"\nEnter your {interactive_game.num_letters}-letter guess (or 'quit'): ").strip()
            if guess.lower() == 'quit':
                print(f"The word was: {interactive_game.target_word}")
                break
                
            hint, continues = interactive_game.play(guess)
            print(f"Hint: {' '.join(hint)}")
            
        except ValueError as e:
            print(f"Error: {e}")
    
    if interactive_game.game_over:
        interactive_game.show()
        interactive_game.stat()