import re
import json


def parse_structure(path: str):
    """Parse the structure of the evaluation text file."""

    with open(path, "r") as file:
        input_text = file.read()

    parsed_data = []

    # Split input text into individual questions and answers
    qa_pairs = re.split(r'\n---\n', input_text)

    for qa_pair in qa_pairs:
        # Extract Frage, Header Paths, and Antwort using regular expressions
        match_frage = re.search(r'\*\*Frage:\*\* (.+?)\n', qa_pair)
        match_header_paths = re.search(r'\*\*Header Paths:\*\* (\[.+?\])\n', qa_pair)
        match_antwort = re.search(r'\*\*Antwort:\*\* (.+)', qa_pair)

        if match_frage and match_header_paths and match_antwort:
            # Create a dictionary for each parsed question-answer pair
            parsed_data.append({
                "query": match_frage.group(1),
                "header_paths": eval(match_header_paths.group(1)),
                "expected_answer": match_antwort.group(1).strip()
            })

    # write parsed data to a file
    with open("data/eval_set_new.json", "w") as file:
        file.write(json.dumps(parsed_data, indent=4, ensure_ascii=False))


parse_structure("data/eval_set.txt")
