from langchain_openai import OpenAI
from llama_index.core import ServiceContext, download_loader, PromptTemplate, DocumentSummaryIndex, get_response_synthesizer

from oepul_chat.readers.custom_full_pdf_reader import CustomFullPDFReader
from oepul_chat.utils import load_data

# Custom prompt for summarization
summary_prompt = PromptTemplate(
    "Du bist ein System welches Zusammenfassungen von Maßnahmen für Landwirte in Österreich aus dem Programm Österreichischen Programm für umweltgerechte Landwirtschaft kurz OEPUL erstellt.\n"
    "Hier die Informationen zu den ÖPUL Förderungen/ Maßnahmen:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Fasse die Maßnahme zusammen, achte besonders auf die Bedingungen und Förderhöhen.\n"
    "Der Landwirt sollte schnell erfassen können, ob die Maßnahme für ihn in Frage kommt.\n"
    "Zusammenfassung: ")


def create_summary_index():
    # Read in each doc as one file
    full_docs = load_data(
        "data/OEPUL_PDF/", ".pdf", CustomFullPDFReader())

    # Use chatgpt model
    chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo-1106")

    # Create service context
    service_context = ServiceContext.from_defaults(llm=chatgpt)

    # Create response synthesizer which summarizes the nodes with custom prompt
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        summary_template=summary_prompt
    )

    # Create document summary index
    doc_summary_index = DocumentSummaryIndex.from_documents(
        documents=full_docs,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        show_progress=True,
    )

    # Persist it for later use
    doc_summary_index.storage_context.persist(persist_dir="indices/summary_index/")
