from llama_index import (
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
)
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from llama_index.llms import ChatMessage, MessageRole

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


completion_prompt = PromptTemplate(
    "Du bist ein Supportsystem für Landwirte in Österreich und bekommst Informationen zum Österreichischen Programm für umweltgerechte Landwirtschaft kurz OEPUL.\n"
    "Anhand dieser Informationen sollst du Landirten helfen Entscheidungen zu treffen so dass sie Förderungen aus dem ÖPUL Programm bekommen.\n"
    "Hier die Informationen zu den ÖPUL Förderungen:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Angesichts der Kontextinformationen und ohne Vorwissen beantworte die Frage:\n"
    "Frage: {query_str}\n"
    "Antwort: ")


chat_prompt = PromptTemplate(
    """\
    "Du bist ein Supportsystem für Landwirte in Österreich und bekommst Informationen zum Österreichischen Programm für umweltgerechte Landwirtschaft kurz OEPUL.\n"
    "Anhand dieser Informationen sollst du Landirten helfen Entscheidungen zu treffen so dass sie Förderungen aus dem ÖPUL Programm bekommen.\n"
    Du bekommst eine Unterhaltung (zwischen Mensch und Assistent) und eine weitere Nachfrag, 
    Anhand des Kontextes formuliere eigenständig eine Frage die das Informationsbedürfniss des Nutzer besser beschreibt.

<Gesprächs Historie>
{chat_history}

<Nachfrage>
{question}

<Neue eigenständige Frage>
""")

custom_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="Du bist ein Supportsystem für Landwirte in Österreich und bekommst Informationen zum Österreichischen Programm für umweltgerechte Landwirtschaft kurz OEPUL.\n"
        "Anhand dieser Informationen sollst du Landirten helfen Entscheidungen zu treffen so dass sie Förderungen aus dem ÖPUL Programm bekommen.",),
    ChatMessage(role=MessageRole.ASSISTANT, content="Verstanden, wie kann ich dir behilflich sein?."),]


def rag_query(query_str: str, index_path: str, k=8):

    print('Loading index from storage...')

    # Rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_path)

    # Load index from the storage context
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(
        summary_template=completion_prompt, similarity_top_k=k, streaming=True)

    print(f'\nFrage:\n{query_str}\n')

    resp = query_engine.query(query_str)

    print(f'Antwort:\n{resp}\n')

    return resp


def rag_chat(service_context, index_path: str = "indices/normal_index/"):

    print('Loading index from storage...')

    # Rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_path)

    # Load index from the storage context
    index = load_index_from_storage(storage_context)

    chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        verbose=True,
        chat_prompt=chat_prompt,
        context_prompt=completion_prompt,
        service_context=service_context,
        similarity_top_k=8,
        custom_chat_history=custom_chat_history)

    chat_engine.chat_repl()


def summary_query(query_str: str):

    # load summaries txt
    with open('data/ .txt', 'r') as f:
        summaries = f.read()

    print(f'\nFrage:\n{query_str}\n')

    messages = [
        SystemMessage(
            content=f"""
    Du bist ein Supportsystem für Landwirte in Österreich und bekommst Informationen zum Österreichischen Programm für umweltgerechte Landwirtschaft kurz OEPUL. Du bekommst Zusammenfassungen verschiedener Föderprogramme und empfiehlst den Nutzer dann passender Förderprogamme.

---- 

Bitte gebe  deine Antwort im folgenden format:
Förderprogrammausgabe: ["Tierwohl-Stallhaltung-Rinder", "Tierwohl-Weide"]

Erläuterung:
    Tierwohl-Stallhaltung-Rinder: Die Maßnahme fördert ... erhalten.
    Tierwohl-Weide: Diese Maßnahme fördert die Weidehaltung ... mehreren Teilnahmejahren möglich.

---- 


Föderprogramme:\n {summaries}\n\n"""
        ),
        HumanMessage(
            content=f"""Frage: {query_str}"""
        ),
    ]

    # completion api
    chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
    print(chat(messages).content)
