from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer

qa_prompt = PromptTemplate(
    "Du bist ein Supportsystem für Landwirte in Österreich und bekommst Informationen zum Österreichischen Programm für umweltgerechte Landwirtschaft kurz OEPUL.\n"
    "Anhand dieser Informationen sollst du Landirten helfen Entscheidungen zu treffen so dass sie Förderungen aus dem ÖPUL Programm bekommen.\n"
    "Hier die Informationen zu den ÖPUL Förderungen:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Angesichts der Kontextinformationen und ohne Vorwissen beantworte die Frage:\n"
    "Frage: {query_str}\n"
    "Antwort: "
)


class RAGOEPULStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    llm: OpenAI

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response)
