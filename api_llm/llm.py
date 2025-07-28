from google import genai

class LLM(genai.Client):
    
    def translet(self, text):
        client = genai.Client(api_key="AIzaSyCmRWQCL2-yvRr4FkvEmSqxmZBPVeLWZXw")

        response = client.models.generate_content(
            model="gemini-2.5-flash", contents="""                        "You are a helpful assistant for manga translation. "
                        "You will be given a word in English and you need to translate it into Turkish. "
                        "You will only return the translated word, nothing else. "
                        "You not translate special names, such as manga names, character names, etc. "
                        "You fix break lines in the text. for example: THIS 15 17 -> This is it "
        """ + "Translate this word: " + text
        )
        return response.text