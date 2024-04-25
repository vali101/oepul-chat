import argparse
import logging
from dotenv import load_dotenv
from llama_index import ServiceContext
from llama_index.embeddings import OpenAIEmbedding
from langchain_openai import OpenAI
from oepul_chat import download_data, create_summary_index, create_normal_index, rag_query, rag_chat, evaluate_retriever, summary_query, generate_responses, evaluate_responses
import warnings
warnings.filterwarnings("ignore")

# Get api key from .env file
load_dotenv()

# set_global_service_context(service_context)


def main():
    parser = argparse.ArgumentParser(description="Chat with OEPUL chatbot")
    parser.add_argument("--download", action="store_true",
                        help="Download data necessary for the project.")
    parser.add_argument("--build-summary-index", action="store_true", help="Create summary index")
    parser.add_argument("--build-normal-index", action="store_true", help="Create normal index")
    parser.add_argument("--build-oepul-base-reader-index", action="store_true",
                        help="Create only OEPUL base reader index")
    parser.add_argument("--evaluate_retriever", action="store_true", help="Evaluate retriever")
    parser.add_argument("--generate_responses", action="store_true", help="Evaluate response")
    parser.add_argument("--evaluate_responses", action="store_true", help="Evaluate response")
    parser.add_argument("--combinations", action="store_true",
                        help="Evaluate chatbot with different combinations")
    parser.add_argument("--build-indices", action="store_true", help="Create all indices: ")
    parser.add_argument("--query", type=str, help="Query chatbot")
    parser.add_argument("--summary-query", type=str, help="Query summaries.")
    parser.add_argument("--chat", action="store_true", help="Chat with chatbot")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--file-path", type=str, help="File path.")
    parser.add_argument("--k", type=int, help="Number of docs retrieved. Default is 6.")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger('llama_index').setLevel(logging.DEBUG)

    if args.download:
        download_data("data/")

    if args.build_summary_index:
        create_summary_index()

    if args.build_normal_index:
        create_normal_index(only_oepul=True)

    if args.build_oepul_base_reader_index:
        create_normal_index(only_oepul=True, only_oepul_base_reader=True)

    if args.build_indices:
        create_summary_index()
        create_normal_index(only_oepul=True)
        create_normal_index(only_oepul=True, only_oepul_base_reader=True)

    if args.chat:
        rag_chat(service_context=service_context)

    if args.evaluate_retriever:
        if args.combinations:
            evaluate_retriever(combinations=True)
        elif args.file_path and args.k:
            evaluate_retriever(args.file_path, args.k)
        elif args.file_path:
            evaluate_retriever(args.file_path)
        elif args.k:
            evaluate_retriever(k=args.k)
        else:
            evaluate_retriever()

    if args.generate_responses:
        generate_responses()

    if args.evaluate_responses:
        evaluate_responses()

    if args.query:
        if args.k:
            rag_query(args.query, "indices/oepul_index/", args.k)
        else:
            rag_query(args.query, "indices/oepul_index/")

    if args.summary_query:
        summary_query(args.summary_query)


if __name__ == "__main__":
    main()
