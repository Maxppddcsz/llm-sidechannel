import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

# Input file path
input_file = r"book_sum_sort\\test.jsonl"

# Output directories
dir_1_100 = "test-10000-train-pdf/1-100"
dir_101_200 = "test-10000-train-pdf/101-200"
dir_201_300 = "test-10000-train-pdf/201-300"
dir_301_400 = "test-10000-train-pdf/301-400"
dir_401_500 = "test-10000-train-pdf/401-500"

# Ensure directories exist
os.makedirs(dir_1_100, exist_ok=True)
os.makedirs(dir_101_200, exist_ok=True)
os.makedirs(dir_201_300, exist_ok=True)
os.makedirs(dir_301_400, exist_ok=True)
os.makedirs(dir_401_500, exist_ok=True)

def count_words(text):
    """Count the number of words in a text."""
    return len(text.split())

def save_as_pdf(text, file_path):
    """Save text as a PDF using reportlab."""
    # Create a document template with letter size
    doc = SimpleDocTemplate(file_path, pagesize=letter)

    # Prepare styles for text
    styles = getSampleStyleSheet()
    style_normal = styles["Normal"]

    # Convert the text to a Paragraph, which supports line breaks and formatting
    paragraph = Paragraph(text, style_normal)

    # Build the PDF document
    doc.build([paragraph])

# Read input data
with open(input_file, "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

print(f"Total number of entries: {len(data)}")

# Process and save files
for idx, entry in enumerate(data):
    input_text = entry["input"]
    word_count = count_words(input_text)

    # Ensure word count is between 10000 and 15000
    while word_count < 10000:
        input_text += " " + entry["input"]
        word_count = count_words(input_text)

    # Trim text if it exceeds 15000 words
    if word_count >= 10000:
        words = input_text.split()
        input_text = " ".join(words[:10000])

    # Determine the directory based on the index
    if 0 <= idx < 100:
        save_dir = dir_1_100
    elif 100 <= idx < 200:
        save_dir = dir_101_200
    elif 200 <= idx < 300:
        save_dir = dir_201_300
    elif 300 <= idx < 400:
        save_dir = dir_301_400
    elif 400 <= idx < 500:
        save_dir = dir_401_500
    else:
        continue  # Skip if idx >= 500 (or modify to other behavior)

    # Save as PDF
    file_name = f"{idx + 1:06}_doc.pdf"  # Use 1-based index for file naming
    file_path = os.path.join(save_dir, file_name)

    # Print log of current processing file
    print(f"Processing file {file_name}...")

    save_as_pdf(input_text, file_path)

print("Files successfully created and saved as PDF.")
