import streamlit as st
import requests

st.set_page_config(page_title="LinkedIn Post Generator", page_icon="💼")

st.title("💼 LinkedIn Post Generator")
st.write("Generate professional LinkedIn posts using AI.")

# -------------------------------
# User Inputs
# -------------------------------
topic = st.text_input("Enter the topic", placeholder="e.g., AI in Healthcare")
language = st.selectbox("Select language", ["English", "Spanish", "Bengali"])
sentences_per_paragraph = st.number_input("Sentences per paragraph", min_value=1, max_value=5, value=5)
paragraphs = st.number_input("Number of paragraphs", min_value=1, max_value=5, value=1)
tone = st.selectbox("Select tone", ["Professional", "Friendly", "Inspirational"])
use_emojis = st.checkbox("Include emojis", value=False)

# -------------------------------
# Generate Post Button
# -------------------------------
if st.button("Generate Post"):
    if not topic.strip():
        st.warning("Please enter a topic!")
    else:
        payload = {
            "topic": topic,
            "language": language,
            "sentences_per_paragraph": sentences_per_paragraph,
            "tone": tone,
            "paragraphs": paragraphs,
            "use_emojis": use_emojis
        }

        try:
            response = requests.post("https://linkdin-post-generator.onrender.com/generate_post", json=payload)
            response.raise_for_status()
            data = response.json()
            post_text = data.get("post", "No post returned.")

            st.subheader("Generated LinkedIn Post:")
            st.write(post_text)

        except Exception as e:
            st.error(f"Error generating post: {e}")
