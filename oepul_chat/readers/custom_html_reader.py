import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from llama_index import SimpleDirectoryReader
from llama_index.readers.base import BaseReader
from llama_index.schema import Document, NodeRelationship, RelatedNodeInfo
from llama_index.readers.file.docs_reader import DocxReader, HWPReader, PDFReader
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class CustomHTMLReader(BaseReader):
    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:

        # file = "data/BIO_Austria/aktueller-planungsstand-zu-bio-im-oepul-2023.html"
        filename = file.name

        # Load article html
        with open(file, 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        article_soup = soup.find('article')

        docs = None
        # try:
        docs = html_to_llama_docs(article_soup, filename)
        # except Exception as e:
        #     print("Failed to convert HTML to dictionary.")

        return docs


# # Function to convert HTML article element to structured dictionary
# def html_to_dict(element):
#     result = {}
#     for child in element.children:
#         if child.name is not None:
#             if child.name in ['h1', 'h2', 'h3']:
#                 result[child.name] = child.get_text().strip()
#             elif child.name == 'p':
#                 if 'text' not in result:
#                     result['text'] = []
#                 result['text'].append(child.get_text().strip())
#             elif child.name in ['ul', 'ol']:
#                 if 'lists' not in result:
#                     result['lists'] = []
#                 current_list = []
#                 for li in child.find_all('li'):
#                     current_list.append(li.get_text().strip())
#                 result['lists'].append({child.name: current_list})
#             elif child.name == 'div':
#                 if 'divs' not in result:
#                     result['divs'] = []
#                 result['divs'].append(html_to_dict(child))
#     return result


# Function to convert HTML article element to a list of llama index Documents
def html_to_llama_docs(article_soup, filename, header_path=[]):
    result = []
    for child in article_soup.children:
        if child.name is not None:
            tag_type = child.name
            if tag_type in ['h1', 'h2', 'h3']:
                header_path.append(child.get_text().strip())
            elif tag_type == 'p':

                result.append(Document(
                    text=child.get_text().strip(),
                    metadata={
                        'File Name': filename,
                        "Content Type": "text",
                        'Header Path': '/'.join(header_path),
                        'tag': tag_type
                    }
                ))
                # result.append({'header_path': '/'.join(header_path),
                #               'tag': tag_type, 'content': child.get_text().strip()})
            elif tag_type in ['ul', 'ol']:
                current_list = []
                for li in child.find_all('li'):
                    current_list.append(li.get_text().strip())
                result.append(Document(
                    text=str(current_list),
                    metadata={
                        'File Name': filename,
                        "Content Type": "text",
                        'Header Path': '/'.join(header_path),
                        'tag': tag_type
                    }
                ))
                # result.append({'header_path': '/'.join(header_path),
                #               'tag': tag_type, 'content': current_list})
            elif tag_type == 'div':
                result.extend(html_to_llama_docs(
                    child, header_path.copy()))
    return result
