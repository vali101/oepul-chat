from langchain_openai import OpenAI
from llama_index.node_parser import SentenceSplitter
from llama_index import (
    download_loader,
    VectorStoreIndex,
)

from oepul_chat.readers.custom_pdf_reader import CustomPDFReader
from oepul_chat.readers.custom_html_reader import CustomHTMLReader
from oepul_chat.utils import load_data


def create_normal_index():
    PDFReader = download_loader("PDFReader")

    # load all official OEPUL docs with custom PDF reader
    oepul_official_docs = load_data("data/OEPUL_PDF/", ".pdf", CustomPDFReader())
    # # load html guide from BIO Austria
    bio_austria_guide = load_data("data/BIO_AUSTRIA", ".html", CustomHTMLReader())
    # # Load rest of AMA docs with simple pdf reader
    ama_official_docs = load_data("data/AMA", ".pdf", PDFReader())

    # merge documents lists
    # [oepul_official_docs, ama_official_docs, bio_austria_guide]  #
    docs_list = [oepul_official_docs, bio_austria_guide, ama_official_docs]
    documents = [doc for docs in docs_list
                 for doc in docs]

    # Define node parser
    node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=20)
    # node_parser = SimpleNodeParser.from_defaults(
    #     text_splitter=TokenTextSplitter(chunk_size=800, chunk_overlap=20)
    # )

    # Parse nodes from docs
    nodes = node_parser.get_nodes_from_documents(documents)

    # Hide certain metadata form llm and embed
    for node in nodes:
        node.excluded_llm_metadata_keys = ["Content Type", "page_label"]
        node.excluded_embed_metadata_keys = ["Content Type"]

    index = VectorStoreIndex(nodes)

    index.storage_context.persist(persist_dir="indices/normal_index/")
