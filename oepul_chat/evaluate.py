from .query_engine import load_index
from .utils import get_header_path_mappings, get_ids_of_header_path
from llama_index.evaluation import RetrieverEvaluator
import json


def evaluate_retriever():
    # Load only OEPUL index
    index = load_index('indices/oepul_index/')

    # open eval set
    with open('data/eval_set_new.json', 'r', encoding='utf-8') as f:
        eval = json.load(f)

    # Setup retriever and evaluator
    retriever = index.as_retriever(similarity_top_k=10)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )

    documents = index.docstore.docs
    header_path_mappings = get_header_path_mappings(documents)

    for item in eval:
        query = item['query']
        expected_header_paths = item['header_paths']
        expected_answer = item['expected_answer']

        true_doc_ids = []
        # check if header_paths empty
        if not expected_header_paths:
            continue

        for header_path in expected_header_paths:
            true_doc_ids.extend(get_ids_of_header_path(header_path, header_path_mappings))

        eval_result = retriever_evaluator.evaluate(
            query=query, expected_ids=true_doc_ids
        )

        print(f'{eval_result}')
