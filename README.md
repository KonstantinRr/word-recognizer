UNIVERSITY OF GRONINGEN - Neural Networks

Konstantin Rolf - S3750558,
Kacper Bielewicz - S3745791,
Nicholas Koundouros - S3726444,
Daniel Aboo - S3472183

# Word-Recognizer
A simple NN word recognizer based on the EMNIST dataset. The model is capable of predicting combinations of letters that form words.

## Run the project
Run the project using the simple command line interface:

```{bash}
git clone https://github.com/KonstantinRr/word-recognizer && cd word-recognizer
pip install -r requirements.txt

# Trains the model for 20 epochs
python train --epochs 20

# Predicts a word using the trained dataset
python predict --weights training/check
```
