# backend/agent.py
import os
from langchain.llms.base import LLM
from dotenv import load_dotenv
import os
from pathlib import Path
from groq import Groq


# Explicitly point to .env in the project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class GroqLLM(LLM):
    """Custom LLM wrapper using Groq SDK for non-streaming completions"""
    
    @property
    def _llm_type(self):
        return "groq-llm"
    
    def _call(self, prompt: str, stop=None) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env!")
        
        client = Groq(api_key=api_key)
        
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=600,
            stream=False
        )
        
        
        result_text = completion.choices[0].message.content.strip()
        
        return result_text

# -------------------------------
# Agent function
# -------------------------------
def generate_linkedin_post(topic: str, 
                           language: str, 
                           sentences_per_paragraph: int, 
                           tone: str = "Professional",
                           paragraphs: int = 1,
                           emojis: bool = False) -> str:
    
    llm = GroqLLM() 
    
    emoji_instruction = "Include some relevant emojis in the paragraph." if emojis else "Do not include any emojis."

    # Advanced prompt with example
    prompt = f"""
You are a professional LinkedIn content writer.
Write a LinkedIn post about "{topic}" in {language}, approximately {sentences_per_paragraph} sentences per paragraph with a {tone} tone.

The post should:
- Include 1-5 relevant hashtags keeping them in a separate line at the end
- Keep sentences concise and clear
- Be suitable for a LinkedIn audience
- {emoji_instruction}

Generate {paragraphs} distinct paragraphs for the topic "{topic}"
"""

    # response = llm.invoke(prompt)
    # post_text = response.content if hasattr(response, "content") else str(response)
    response= llm(prompt)
    post_text = response or ""
    if not isinstance(post_text, str):
        post_text = str(post_text)
    # print("✅ Generated post:", post_text)
    return post_text

if __name__ == "__main__":
    post = generate_linkedin_post("AI in Healthcare", "English", 5)
    print(post)
