import requests

payload = {
    "topic": "AI in Healthcare",
    "language": "English",
    "sentences_per_paragraph": 5,
    "paragraphs": 2,
    "tone": "Professional",
    "use_emojis": True
}

response = requests.post("http://127.0.0.1:8000/generate_post", json=payload)
print(response.json())
