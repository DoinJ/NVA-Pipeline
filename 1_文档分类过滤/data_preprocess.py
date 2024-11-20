import os
import re
import json
from collections import defaultdict
import csv
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import re
from html import unescape

# Set the path to the data directory
data_dir = './2023'

# Set the path to the output CSV file
output_file = 'source_counts.csv'

# Get the list of txt files in the data directory
txt_files = [file for file in os.listdir(data_dir) if file.endswith('.txt')]

# Initialize variables for total news count and file number
total_news_count = 0
file_number = 1

# Set the path to the SimHei font file
font_path = 'simhei.ttf'  # Replace with the actual path to the SimHei font file

# Create a FontProperties object with the SimHei font
font_prop = FontProperties(fname=font_path)

# Process the txt files one by one
for file_name in txt_files:
    file_path = os.path.join(data_dir, file_name)
    
    # Initialize a dictionary to store the count of news articles per source for the current file
    source_count = defaultdict(int)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Extract news articles using regular expressions
        news_articles = re.findall(r'{.*?}', content, re.DOTALL)
        
        for article in news_articles:
            try:
                # Preprocess the article string
                article = re.sub(r"'", '"', article)
                article = unescape(article)
                article = re.sub(r'<[^>]*>', '', article)
                article_dict = json.loads(article, strict=False)
                source = article_dict.get('source')
                if source:
                    source_count[source] += 1
                    total_news_count += 1
            except json.JSONDecodeError:
                # Handle JSON parsing errors
                # print(f"Error parsing JSON: {article}")
                # Debugging: Print the problematic article
                # print("Problematic article:")
                # print(article)
                # Debugging: Attempt to parse the article again and print the error
                try:
                    json.loads(article)
                except json.JSONDecodeError as e:
                    print("JSON parsing error:")
                    print(str(e))
                    break
    
    # Update the CSV file with the source counts for the current file
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if file_number == 1:
            writer.writerow(['Source', 'Count'])
        for source, count in source_count.items():
            writer.writerow([source, count])
    
    # Print the summary for the current file
    print(f"File {file_number} summary:")
    print(f"Number of news articles processed: {sum(source_count.values())}")
    print(f"Total news articles processed so far: {total_news_count}")
    print("Top 5 sources in the current file:")
    for source, count in sorted(source_count.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{source}: {count}")
    print()
    
    file_number += 1
    
    # Free the memory by clearing the source_count dictionary
    source_count.clear()

# Print the final summary
print("Final summary:")
print(f"Total number of news articles processed: {total_news_count}")

# Read the source counts from the CSV file
source_count = defaultdict(int)
with open(output_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        source, count = row
        source_count[source] += int(count)

# Sort the sources by count in descending order
sorted_sources = sorted(source_count.items(), key=lambda x: x[1], reverse=True)

# Get the top 5 sources
top_sources = sorted_sources[:5]

# Print the top 5 sources
print("Top 5 sources overall:")
for source, count in top_sources:
    print(f"{source}: {count}")

'''
# Extract the source names and counts
sources = [source[0] for source in top_sources]
counts = [source[1] for source in top_sources]

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(sources, counts)
plt.xlabel('News Source', fontproperties=font_prop)
plt.ylabel('Number of News Articles', fontproperties=font_prop)
plt.title('Top 5 News Sources', fontproperties=font_prop)
plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
plt.tight_layout()
plt.show()
'''