# Enhancing Propaganda Detection with Open-Source Language Models: A Comparative Study

## Introduction

This repository contains the code and data for the project "Enhancing Propaganda Detection with Open-Source Language Models: A Comparative Study." The aim of this project is to evaluate the performance of various open-source language models in detecting propaganda in text data. The study involves the use of different models and techniques to compare their effectiveness in identifying propaganda.

## Repository Structure

The repository is organized as follows:

- **1_model_out_of_the_box**: Contains the script `oob_tester.py` for the initial evaluation of language models without any fine-tuning.
- **2_model_with_modelfile**: Includes the script `tester_modelfile.py` and the `Modelfile` for models that have been fine-tuned using modelfile and ollama framework.
- **3_model_with_inference**: Houses the scripts used for performing inference with the fine-tuned models, including `Tester_inference.py`, `ingest.py`, `constants.py`, and other supporting files and folders.
- **Evaluator.py**: The main script for evaluating the performance of different models and generating comparative results.
- **dev-articles**: Directory containing the development set of articles used for training and evaluation.
- **dev-labels**: Contains the labels for the development articles.
- **results_inference**: Stores the results obtained from inference using the models.
- **results_modelfile**: Contains the results from models fine-tuned with specific training files.
- **results_oob**: Stores the results from out-of-the-box models without any fine-tuning.

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/AlexParlam/Propaganda-Detection-Comparative-Study.git
   cd Propaganda-Detection-Comparative-Study
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Usage

1. **Out-of-the-Box Model Evaluation**:
   Navigate to the `1_model_out_of_the_box` directory and run the evaluation script `oob_tester.py`.

   ```bash
   cd 1_model_out_of_the_box
   python oob_tester.py
   ```

2. **Model with Modelfile**:
   Navigate to the `2_model_with_modelfile` directory and run the fine-tuning and evaluation scripts.

   ```bash
   cd 2_model_with_modelfile
   python tester_modelfile.py
   ```

3. **Inference with Fine-Tuned Models**:
   Navigate to the `3_model_with_inference` directory and run the inference scripts.

   ```bash
   cd 3_model_with_inference
   python Tester_inference.py
   ```

4. **Evaluator**:
   Use the `Evaluator.py` script to compare the results of different models.

   ```bash
   python Evaluator.py
   ```

## Results

The results of the evaluations and inferences are stored in their respective directories:
- `results_oob`: Results from out-of-the-box models.
- `results_modelfile`: Results from models fine-tuned with modelfile (by ollama).
- `results_inference`: Results from model with integrated LangChain technology and the all-MiniLM-L6-v2 embedding model.


For any questions or inquiries, please contact Oleksandr Lytvyn
