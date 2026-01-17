# Genetic Algorithm Vigenère Cryptanalysis

This project applies a **genetic algorithm** to perform **cryptanalysis of the Vigenère cipher**. The goal is to automatically recover the encryption key and plaintext from a ciphertext without prior knowledge of the key.

Candidate keys are evolved using a population-based genetic algorithm with selection, crossover, and mutation. Each candidate is evaluated using a fitness function based on statistical properties of English text, guiding the search toward increasingly accurate decryptions.

The project demonstrates how evolutionary algorithms can be used to search large, non-linear solution spaces where exhaustive search is impractical.

## Features
- Genetic algorithm for key discovery
- Fitness evaluation using language statistics
- Selection, crossover, and mutation operators
- Automated Vigenère cipher decryption

## Techniques & Concepts
- Genetic algorithms
- Evolutionary search
- Cryptanalysis
- Fitness function design
- Heuristic optimization

## Technologies Used
- Python

## Project Context
Developed as part of **COSC 3P71 – Artificial Intelligence** at Brock University.
