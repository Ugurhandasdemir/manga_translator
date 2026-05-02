import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from google import genai
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = (
    "You are a professional English (en) to Turkish (tr) translator. "
    "Your goal is to accurately convey the meaning and nuances of the original English text "
    "while adhering to Turkish grammar, vocabulary, and cultural sensitivities. "
    "Produce only the Turkish translation, without any additional explanations or commentary."
)


class LLM:
    def gemini(self, text: str) -> str:
        gemini_api_key = os.getenv("gemini_api_key")
        if not gemini_api_key:
            raise RuntimeError("gemini_api_key bulunamadı (.env).")

        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [SYSTEM_PROMPT + "\n\nTranslate:\n" + text]}],
        )
        return str(response.candidates[0].content.parts[0]).strip()

    def openai(self, text: str) -> str:
        openai_api_key = os.getenv("openai_api_key")
        if not openai_api_key:
            raise RuntimeError("openai_api_key bulunamadı (.env).")

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
        return str(answer).strip()