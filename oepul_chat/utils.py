from llama_index.core import SimpleDirectoryReader


def get_ids_of_header_path(header_path, header_path_mappings):
    return header_path_mappings[header_path]


def load_data(filepath, filetype, reader):
    """Load markdown docs from a directory, excluding all other file types."""
    print(f'loading data... {filepath}')
    loader = SimpleDirectoryReader(
        input_dir=filepath,
        file_extractor={filetype: reader},
        recursive=True
    )

    data = loader.load_data()

    # print short summary
    print("Loaded {} documents".format(len(data)))
    print("First document metadata: {}".format(data[1].metadata))
    print("First document text: {}".format(data[1].text[0:80]))

    return data


def get_header_path_mappings(documents):
    header_paths = dict()
    for doc in documents.values():
        header_path = doc.metadata['Header Path'] if 'Header Path' in doc.metadata else None
        if not header_path in header_paths:
            header_paths[header_path] = [doc.id_]
        else:
            header_paths[header_path].append(doc.id_)

    return header_paths
