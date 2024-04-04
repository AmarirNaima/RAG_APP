from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    
    ## the model  is loaded here, so that it can be shared across multiple users in a single session
    model = Ollama(model="mistral")
    
    ## the prompte  template defines how to generate prompts and parse responses for this chatbot
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable Scientise who provides accurate and eloquent answers to data Analyst or Data Scientis questions.",
            ),
            ("human", "{question}"),
        ]
    )
    
    ## Setting the chain
    runnable = prompt | model | StrOutputParser()
    
    ## Setting the user_session  
    # decorator makes sure that each message from a user  gets processed individually by the same instance of the class
    
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
