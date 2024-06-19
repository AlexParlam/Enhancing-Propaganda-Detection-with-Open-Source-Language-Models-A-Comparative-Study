# evaluating recall, precision and F1 score.
# saves results in the main folder
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define mapping of fuzzy responses to propaganda techniques with more variations
technique_mapping = {
    "Appeal to Authority": ["Appeal_to_Authority"],
    "Appeal to fear-prejudice": ["Appeal_to_fear-prejudice"],
    "Bandwagon": ["Bandwagon,Reductio_ad_hitlerum"],
    "Black-and-White Fallacy": ["Black-and-White_Fallacy"],
    "Causal Oversimplification": ["Causal_Oversimplification"],
    "Doubt": ["Doubt"],
    "Exaggeration": ["Exaggeration,Minimisation"],
    "Minimization": ["Exaggeration,Minimisation"],
    "Flag-Waving": ["Flag-Waving"],
    "Loaded Language": ["Loaded_Language"],
    "Name-Calling": ["Name_Calling,Labeling"],
    "Labeling": ["Name_Calling,Labeling"],
    "Repetition": ["Repetition"],
    "Slogans": ["Slogans"],
    "Thought-terminating Cliches": ["Thought-terminating_Cliches"],
    "Whataboutism": ["Whataboutism,Straw_Men,Red_Herring"],
    "Straw Men": ["Whataboutism,Straw_Men,Red_Herring"],
    "Red Herring": ["Whataboutism,Straw_Men,Red_Herring"]
}

# Function to map fuzzy responses to techniques
def map_fuzzy_to_techniques(response):
    techniques = []
    response_lines = response.split('\n')
    for line in response_lines:
        for key, value in technique_mapping.items():
            if key.lower() in line.lower().replace("/", " ").replace("-", " "):  # Case insensitive matching, handle slashes and hyphens
                techniques.extend(value)
    return list(set(techniques))  # Remove duplicates

# Function to load files from a directory
def load_files(file_list, directory):
    data = {}
    for filename in file_list:
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):  # Ensure it is a file
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                try:
                    content = file.read()
                    data[filename] = content
                except Exception as e:
                    logging.warning(f"Could not read file {filename}: {e}")
    return data

# Function to parse true labels
def parse_true_labels(data):
    labels = {}
    for filename, content in data.items():
        if content.strip():  # Non-empty file
            labels[filename] = content.strip().split('\n')
        else:  # Empty file, no propaganda
            labels[filename] = ["no_propaganda"]
    return labels

# Function to clean and sanitize labels
def clean_labels(labels, known_labels):
    cleaned_labels = []
    for label in labels:
        cleaned_label = ''.join(c for c in label if c.isprintable()).strip()
        if cleaned_label and cleaned_label in known_labels:
            cleaned_labels.append(cleaned_label)
    return cleaned_labels

# Function to evaluate the model's performance and save results to a file
def evaluate_model(true_labels_dir, model_responses_dir):
    # List all files in the directories
    label_files = os.listdir(true_labels_dir)
    response_files = os.listdir(model_responses_dir)

    # Load true labels and model responses
    true_labels = load_files(label_files, true_labels_dir)
    model_responses = load_files(response_files, model_responses_dir)

    # Parse true labels
    known_labels = list(technique_mapping.keys()) + [val for sublist in technique_mapping.values() for val in sublist]
    true_labels_parsed = {filename: clean_labels(labels, known_labels) for filename, labels in parse_true_labels(true_labels).items()}

    # Apply the mapping function to model responses
    mapped_responses = {filename: map_fuzzy_to_techniques(response) for filename, response in model_responses.items()}

    # Handle case where model detects no propaganda
    for filename in mapped_responses:
        if not mapped_responses[filename]:
            mapped_responses[filename] = ["no_propaganda"]

    # Only consider files present in both directories
    common_files = set(true_labels_parsed.keys()).intersection(set(mapped_responses.keys()))

    true_labels_filtered = [true_labels_parsed[file] for file in common_files]
    predicted_labels_filtered = [mapped_responses[file] for file in common_files]

    # Flatten lists for calculating precision, recall, and F1 score
    true_labels_flat = [label for sublist in true_labels_filtered for label in sublist]
    predicted_labels_flat = [label for sublist in predicted_labels_filtered for label in sublist]

    # Define the unique set of labels (for all classes in true and predicted labels)
    unique_labels = list(set(true_labels_flat + predicted_labels_flat))

    # Clean the unique labels
    unique_labels = clean_labels(unique_labels, known_labels)

    # Create binary representation for multi-label evaluation
    true_binary = [[1 if label in labels else 0 for label in unique_labels] for labels in true_labels_filtered]
    predicted_binary = [[1 if label in labels else 0 for label in unique_labels] for labels in predicted_labels_filtered]

    # Log the unique labels for debugging
    logging.info(f"Unique labels: {unique_labels}")
    logging.info(f"True binary: {true_binary}")
    logging.info(f"Predicted binary: {predicted_binary}")

    # Calculate precision, recall, and F1 score
    precision = precision_score(true_binary, predicted_binary, average='macro', zero_division=1)
    recall = recall_score(true_binary, predicted_binary, average='macro', zero_division=1)
    f1 = f1_score(true_binary, predicted_binary, average='macro', zero_division=1)

    # Save results to a text file
    results = f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n"
    with open("evaluation_results.txt", 'w') as file:
        file.write(results)

    return precision, recall, f1

# Example usage
true_labels_dir = "/Users/alexparlam/Desktop/fieldtesting/data/dev-labels"
model_responses_dir = "/Users/alexparlam/Desktop/fieldtesting/results_modelfile"

precision, recall, f1 = evaluate_model(true_labels_dir, model_responses_dir)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
