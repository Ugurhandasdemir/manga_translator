import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from google import genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    """You are a professional English (en) to Turkish (tr) translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Turkish grammar, vocabulary, and cultural sensitivities.
    Produce only the Turkish translation, without any additional explanations or commentary. Please translate the following English text into Turkish:"""
)


class LLM:
    def gemini(self, text):
        gemini_api_key = os.getenv("gemini_api_key")
        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [SYSTEM_PROMPT + "\n\nTranslate:\n" + text],
                }
            ],
        )
        return str(response.candidates[0].content.parts[0]).strip()

    def openai(self, text):
        openai_api_key = os.getenv("openai_api_key")
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content.strip()

    def ollama(self, text: str) -> str:
        template = SYSTEM_PROMPT + "\n\nTranslate:\n{text}\n\nTurkish translation:"
        prompt = ChatPromptTemplate.from_template(template)
        model = OllamaLLM(model="translategemma:4b")
        chain = prompt | model
        answer = chain.invoke({"text": text})
        return answer.strip()