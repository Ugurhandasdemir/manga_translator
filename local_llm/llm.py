from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

class OllamaChatLLM(OllamaLLM,):
    def answer(self, question: str) -> str:
        system = """ "You are a helpful assistant for manga translation. "
                        "You will be given a word in English and you need to translate it into Turkish. "
                        "You will only return the translated word, nothing else. "
                        "You not translate special names, such as manga names, character names, etc. "
                        "You fix break lines in the text. for example: THIS 15 17 -> This is it " """

        template = """System: {system}
        This word translation: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        model = OllamaLLM(model = "gemma3:12b")

        chain = prompt | model
        

        answer = chain.invoke({
            "system": system,
            "question": question
        })
                
        return answer