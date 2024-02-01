from llama_index import SimpleDirectoryReader


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
