import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

DATA_DIR = "data"

# Helper to read any JSON file
def load_json(file_name):
    with open(os.path.join(DATA_DIR, file_name), "r") as f:
        return json.load(f)

# 1. Profile to natural language
def format_profile(data):
    return [f"""{data['name']} was born in {data['country']} in the year {data['birth_year']} and uses {data['units']} units.
They identify as {data['gender']} and prefer reminders around {data['reminder_time']}.
Health goals: {', '.join(data['health_goals'])}. They like: {', '.join(data['likes'])}."""], "user_profile"

# 2. Wearable data to natural language
def format_wearable(data):
    docs = []
    for entry in data:
        text = f"On {entry['date']}, average heart rate was {entry['avg_heart_rate']} bpm, steps taken: {entry['steps_taken']}, sleep: {entry['sleep_duration_hrs']} hours, mood: {entry['mood']}."
        docs.append(text)
    return docs, "wearable"

# 3. Chat history
def format_chat(data):
    docs = []
    for entry in data:
        text = f"User asked: '{entry['user_msg']}' → Bot replied: '{entry['bot_reply']}' (at {entry['timestamp']})."
        docs.append(text)
    return docs, "chat_history"

# 4. Location data
def format_locations(data):
    docs = []
    for entry in data:
        text = f"Visited {entry['place']} at {entry['time']}."
        docs.append(text)
    return docs, "location_data"

# 5. Custom collection
def format_custom(data):
    docs = []
    for meal in data:
        text = f"On {meal['day']}, had {meal['meal']} (~{meal['cals']} calories): {meal['note']}."
        docs.append(text)
    return docs, "custom_collection"

# Format and combine all documents
all_documents = []

# Chunking setup
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)

# Map formatters to filenames
sources = {
    "user_profile.json": format_profile,
    "wearable.json": format_wearable,
    "chat_history.json": format_chat,
    "location_data.json": format_locations,
    "custom_collection.json": format_custom,
}

# Process each collection
for fname, formatter in sources.items():
    raw_data = load_json(fname)
    texts, source = formatter(raw_data)
    chunks = text_splitter.create_documents(texts)
    for chunk in chunks:
        chunk.metadata = {"source": source}
    all_documents.extend(chunks)

''''
for i, doc in enumerate(all_documents[:5]):
    print(f"\n--- Document {i+1} ---")
    print(f"Source: {doc.metadata['source']}")
    print(doc.page_content)


#print (all_documents)
'''
