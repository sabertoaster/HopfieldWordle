# HopfieldNetwork Wordle
A toy project to leverage the properties of Hopfield Network to converge on a finite set of words within Wordle dictionary.

Wordle format:
- A sensible english word contains 5 letters, notated as: \_ \_ \_ \_ \_ (5 consecutive underslash).
- [\-] will represent the wrongly placed of that character (Yellow)
- [letter] will represent a correct letter at that correct position (in which I will relate to as fixated) (Green)
- [\_] will represent a non-exist letter. (Gray)

For instances: The word I'm looking for is APPLE.
1. OUTER <br>
\_ \_ \_ \- \_
1. PUPPY <br>
\- \_ P \_ \_ 

## Research 
Step 1: Find related papers, ideas and how to config the network

Step 2: Implement a baseline of Hopfield model

## Programming
Step 1: Implement a custom version of Hopfield mechanics

Step 2: Run and assure that it runs well


## Present & Deploy
Step 1: Create a proper blog post explaining all the stuff.

Step 2: Deploy directly on github pages using vanilla javascripts (Converted from python).


## References:
- Game mechanics: https://www.youtube.com/watch?v=KRePf4yJz-g&ab_channel=Mehul-Codedamn
- Wordle Engine: https://github.com/yanbenjamin/Wordle-Engine
- 5 words list: https://gist.githubusercontent.com/scholtes/94f3c0303ba6a7768b47583aff36654d/raw/73f890e1680f3fa21577fef3d1f06b8d6c6ae318/wordle-La.txt
- 4 words list: https://gist.github.com/raspberrypisig/cc18b0f4fbc0c79ffd667d06adc0a190#file-4-letter-words-processed-new-txt-L5
- Hopfield Network Explained: https://www.tutorialspoint.com/artificial_neural_network/artificial_neural_network_hopfield.htm
- Hopfield Network is All You Need: https://ml-jku.github.io/hopfield-layers/#energy
- Hopfield Network Pytorch: https://github.com/hmcalister/Hopfield-Network-PyTorch