import argparse
import logging
from dotenv import load_dotenv
from llama_index import set_global_service_context
from llama_index import ServiceContext
from llama_index.embeddings import OpenAIEmbedding
from langchain_openai import OpenAI
from oepul_chat import download_data, create_summary_index, create_normal_index, rag_query, rag_chat

# Get api key from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('llama_index').setLevel(logging.DEBUG)

# Set openai as global service context
# llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0, max_tokens=256)
embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(
    # llm=llm,
    embed_model=embed_model,
)

# set_global_service_context(service_context)


def main():
    parser = argparse.ArgumentParser(description="Chat with OEPUL chatbot")
    parser.add_argument("--download", action="store_true", help="Download data")
    parser.add_argument("--build-summary-index", action="store_true", help="Create summary index")
    parser.add_argument("--build-normal-index", action="store_true", help="Create normal index")
    parser.add_argument("--build-indices", action="store_true", help="Create all indices: ")
    parser.add_argument("--query", type=str, help="Query chatbot")
    parser.add_argument("--chat", action="store_true", help="Chat with chatbot")

    args = parser.parse_args()

    if args.download:
        download_data("data/")

    if args.build_summary_index:
        create_summary_index()

    if args.build_normal_index:
        create_normal_index()

    if args.build_indices:
        create_summary_index()
        create_normal_index()

    if args.chat:
        rag_chat(service_context=service_context)

    if args.query:
        print(rag_query(args.query))


if __name__ == "__main__":
    main()
