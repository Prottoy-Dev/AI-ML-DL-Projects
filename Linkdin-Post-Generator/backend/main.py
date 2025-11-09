from fastapi import FastAPI
from pydantic import BaseModel
try:
    from agent import generate_linkedin_post
except ImportError:
    from backend.agent import generate_linkedin_post

app = FastAPI()

class PostRequest(BaseModel):
    topic: str
    language: str
    sentences_per_paragraph: int
    paragraphs: int
    tone: str
    use_emojis: bool = False

@app.post("/generate_post")
def generate_post(request: PostRequest):
    try:
        final_post = generate_linkedin_post(
            topic=request.topic,
            language=request.language,
            sentences_per_paragraph=request.sentences_per_paragraph,
            paragraphs=request.paragraphs,  
            tone=request.tone,
            emojis=request.use_emojis
        )
        return {"post": final_post}
    except Exception as e:
        return {"error": str(e)}
