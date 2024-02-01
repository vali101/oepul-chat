from llama_index import (
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
)
from langchain_openai import OpenAI
from llama_index.llms import ChatMessage, MessageRole

completion_prompt = PromptTemplate(
    "Du bist ein Supportsystem für Landwirte in Österreich und bekommst Informationen zum Österreichischen Programm für umweltgerechte Landwirtschaft kurz OEPUL.\n"
    "Anhand dieser Informationen sollst du Landirten helfen Entscheidungen zu treffen so dass sie Förderungen aus dem ÖPUL Programm bekommen.\n"
    "Hier die Informationen zu den ÖPUL Förderungen:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Angesichts der Kontextinformationen und ohne Vorwissen beantworte die Frage:\n"
    "Frage: {query_str}\
        n"
    "Antwort: ")
completion_prompt = PromptTemplate(
    "Du bist ein Supportsystem für Landwirte in Österreich und bekommst Informationen zum Österreichischen Programm für umweltgerechte Landwirtschaft kurz OEPUL.\n"
    "Anhand dieser Informationen sollst du Landirten helfen Entscheidungen zu treffen so dass sie Förderungen aus dem ÖPUL Programm bekommen.\n"
    "Hier die Informationen zu den ÖPUL Förderungen:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Angesichts der Kontextinformationen sowie der Chat Historie und ohne Vorwissen beantworte die Frage:\n"
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


def rag_query(query_str: str, index_path: str = "indices/normal_index/"):

    print('Loading index from storage...')

    # Rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_path)

    # Load index from the storage context
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(
        summary_template=completion_prompt, similarity_top_k=8, streaming=True)

    print(f'Querying... ({query_str})')

    resp = query_engine.query(query_str)

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
