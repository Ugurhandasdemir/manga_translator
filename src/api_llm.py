import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from google import genai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 

class LLM:
    def gemini(self, text):
        gemini_api_key = os.getenv("gemini_api_key")
        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        "You are a helpful assistant for manga translation. "
                        "You will be given a word in English and you need to translate it into Turkish. "
                        "You will only return the translated word, nothing else. "
                        "You not translate special names, such as manga names, character names, etc. "
                        "You fix break lines in the text. for example: THIS 15 17 -> This is it. "
                        "Translate this word: " + text
                    ]
                }
            ]
        )
        return response.candidates[0].content.parts[0]

    def openai(self, text):
        openai_api_key = os.getenv("openai_api_key")
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a helpful assistant for manga translation. Your task is to translate English text into Turkish. Please follow these instructions: Provide only the Turkish translation, without any extra text, explanations, or conversational fillers. Correct common typos and broken lines; for example, fix "THIS 15 17" to "This is it." Use a natural, contemporary Turkish tone and vocabulary suitable for a modern manga audience and following proper spelling and formatting rules. When handling proper nouns, translate descriptive or well-known names that have a direct and widely accepted Turkish equivalent, such as "God of Death" -> "Ölüm Tanrısı", "King Arthur" -> "Kral Arthur", or "The Holy Relic" -> "Kutsal Emanet." However, do not translate unique names of specific people, places, or brands that lack a standard Turkish equivalent, such as "Goku," "Excalibur," or "Tokyo," and leave them in their original form."""},
                {"role": "user", "content": "Translate this word: " + text}
            ]
        )
        return response.choices[0].message.content
    
    
    def ollama(self, question: str) -> str:
        system = """
        You are a translator for manga dialogues.

        Rules:
        1. Translate from English to Turkish.
        2. Do not translate proper names (manga titles, character names, location names, etc.).
        3. Fix broken words or lines. Example: "THIS 15 17" → "This is it".
        4. Remove any "-" characters from the text unless they are part of a proper name.
        5. Return only the translated text. No explanations, no extra words.
        """
        template = """
        System: {system}
        Translate this: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        model = OllamaLLM(model = "llama3.1:8b")

        chain = prompt | model
        

        answer = chain.invoke({
            "system": system,
            "question": question
        })
                
        return answer