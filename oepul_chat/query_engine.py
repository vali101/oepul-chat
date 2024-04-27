from llama_index.core import PromptTemplate, StorageContext, load_index_from_storage
from langchain_openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole

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


def load_index(index_path: str = "indices/normal_index/"):

    # Rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_path)

    # Load index from the storage context
    index = load_index_from_storage(storage_context)

    return index


def get_query_engine(index_path: str):

    index = load_index(index_path)

    query_engine = index.as_query_engine(
        summary_template=completion_prompt, similarity_top_k=8, streaming=True)

    return query_engine
