import os
import pandas as pd
from markdown2 import Markdown



## Function to create turn subfolder from turn name
def create_turn_subfolder(turn_name):
    # Get current working directory
    cwd = os.getcwd()
    turn_folder = os.path.join(cwd, 'turns')
    # Create turn subfolder
    turn_subfolder = os.path.join(turn_folder, turn_name)
    # Check if turn subfolder exists
    if not os.path.exists(turn_subfolder):
        # Create turn subfolder
        os.makedirs(turn_subfolder)
    # Return turn subfolder
    return turn_subfolder


# Create model.json with turn type and research findings
def create_model_json(turn_subfolder, turn_type, research_findings):
    # Create model.json
    model_json = os.path.join(turn_subfolder, 'model.json')
    # Check if model.json exists
    if not os.path.exists(model_json):
        # Create model.json
        with open(model_json, 'w') as f:
            # Write turn type and research findings to model.json

            pass
    # Return model.json
    return model_json

# Copy uploaded CSV file and rename it to dataset.csv
def copy_uploaded_file(turn_subfolder: str, uploaded_file):
    # Create dataset.csv
    dataset_csv = os.path.join(turn_subfolder, 'dataset.csv')
    # Check if dataset.csv exists
    if not os.path.exists(dataset_csv):
        # Copy uploaded file to dataset.csv
        df = pd.read_csv(uploaded_file)
        df.to_csv(dataset_csv, index=False)
    # Return dataset.csv
    return dataset_csv

# Convert markdown to HTML
def convert_markdown_to_html(markdown: str) -> str:
    # Convert markdown to HTML
    html = Markdown().convert(markdown)
    # Return HTML
    return html
