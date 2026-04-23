import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Define the paths for ham and phishing emails
ham_path = r"C:\Users\12697\OneDrive\Desktop\CS_AI_PYSHING\ham"
phishing_path = r"C:\Users\12697\OneDrive\Desktop\CS_AI_PYSHING\spam"

# Function to clean HTML tags
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text()

# Function to extract links
def extract_links(text):
    links = re.findall(r'http[s]?://\S+', text)
    return len(links)

# Function to preprocess emails
def preprocess_email(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            email_content = file.read()
            print(f"Processing file: {file_path}")
            body_text = clean_html(email_content)
            num_links = extract_links(body_text)
            key_phrases = " ".join([phrase for phrase in ["free", "save", "click here", "urgent", "limited offer"]
                                    if phrase in body_text.lower()])
            processed_text = f"{body_text} {key_phrases} links_count:{num_links}"
        return processed_text
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Load data and labels
data = []
labels = []

print("Starting preprocessing of ham emails...")
# Process ham emails = 0
for filename in os.listdir(ham_path):
    file_path = os.path.join(ham_path, filename)
    processed_text = preprocess_email(file_path)
    data.append(processed_text)
    labels.append(0)  # Label for ham emails
print("Finished processing ham emails.")

print("Starting preprocessing of phishing emails...")
# Process phishing emails = 1
for filename in os.listdir(phishing_path):
    file_path = os.path.join(phishing_path, filename)
    processed_text = preprocess_email(file_path)
    data.append(processed_text)
    labels.append(1)  # Label for phishing emails
print("Finished processing phishing emails.")


print("Creating DataFrame...")
# Create DataFrame
df = pd.DataFrame({'processed_text': data, 'label': labels})

# Save data to a pickle file
output_path = r"C:\Users\12697\OneDrive\Desktop\CS_AI_PYSHING\preprocessed_emails.pkl"
print(f"Saving preprocessed data to {output_path}...")
df.to_pickle(output_path)
print("Data successfully saved.")
