from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate ,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_community.llms import Ollama

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def load_model():
    llm = Ollama(
        model="mistral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm

## 1. Load the language model
LLM = Ollama(model="orca-mini", temperature=0)

llm = load_model()


##  2. Define prompt templates for human and system messages

chat_Prompt = ChatPromptTemplate.from_messages([
  SystemMessagePromptTemplate.from_template("Tu es un assistant qui renvoi une liste d'animaux qui ont une couleur donnée ,vous pouver me donné la réponse sous format Json"),
  HumanMessagePromptTemplate.from_template ("Quesl animaux ont la couleur {color}?")
    
])

output_parser =JsonOutputParser()

##  3. Generate a response to a given prompt template
## sont utiliser la chain

def get_animaux(color):
    ##Formatage du Promt :
    
    Prompt_messages = chat_Prompt.format_prompt(color=color).to_messages()
    
    ## Appel LLM avec le msg
    resultat_string = llm.invoke(Prompt_messages)
    
    ## Appel The LLm
    return output_parser.invoke(resultat_string)


## avec la chain 
def get_animal(color):
    ## USE the LLM
    chain = chat_Prompt| LLM | output_parser
    ## USE THE llm
    # chain = chat_Prompt | llm | output_parser
    return chain.invoke(color)


print (get_animaux("vert"))
print(get_animal("blan"))  # On appelle la fonction pour tester si ca marche

    
    