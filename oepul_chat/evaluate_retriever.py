from .query_engine import load_index
from .utils import get_header_path_mappings, get_ids_of_header_path
from llama_index.evaluation import RetrieverEvaluator
import json
import pandas as pd


def get_index_and_mappings(index_path: str):
    index = load_index(index_path)
    documents = index.docstore.docs
    header_path_mappings = get_header_path_mappings(documents)
    return index, header_path_mappings


def evaluate_retriever(index_path: str = 'indices/oepul_index/', k=6, combinations=False):
    print(f'Evaluate Retriever from index: {index_path}')

    # open eval set
    with open('data/eval_set_new.json', 'r', encoding='utf-8') as f:
        eval_set = json.load(f)

    top_ks = [2, 4, 6, 8, 10]
    index_paths = ['indices/oepul_index/', 'indices/oepul_index_base_reader/']

    rows = []

    if combinations:
        for index_path in index_paths:
            name = f'{index_path.split("/")[-2]}'
            index, header_path_mappings = get_index_and_mappings(index_path)
            for k in top_ks:
                print(f'Evaluate Retriever from index: {name}, with k={k}')
                rows.append(eval_combination(index, k, eval_set, header_path_mappings, name))

    if not combinations:
        name = f'{index_path.split("/")[-2]}'
        print(f'Evaluate Retriever from index: {name} with k={k}')
        index, header_path_mappings = get_index_and_mappings(index_path)
        print(header_path_mappings)
        rows.append(eval_combination(index, k, eval_set, header_path_mappings, name))

    df = pd.concat(rows)
    print(df)

    df.to_csv('results/eval_retriever_results.csv', index=False)
    df.to_latex('results/eval_retriever_results.tex', index=False)


def eval_combination(index, k, eval_set, header_path_mappings, name):

    # Setup retriever and evaluator
    retriever = index.as_retriever(similarity_top_k=k)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )

    metrics_list = []
    for item in eval_set:
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

        metrics_list.append(eval_result)

    return display_results(name, k, metrics_list)


def display_results(name, k, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    columns = {"retrievers": [name], "k": k, "hit_rate": [hit_rate], "mrr": [mrr]}

    metric_df = pd.DataFrame(columns)

    return metric_df
