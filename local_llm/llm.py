from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

class OllamaChatLLM(OllamaLLM,):
    def answer(self, question: str) -> str:
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