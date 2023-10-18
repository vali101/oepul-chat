import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from llama_index import SimpleDirectoryReader
from llama_index.readers.base import BaseReader
from llama_index.schema import Document, NodeRelationship, RelatedNodeInfo
from llama_index.readers.file.docs_reader import DocxReader, HWPReader, PDFReader
import pdfplumber


logger = logging.getLogger(__name__)


class CustomPDFReader(BaseReader):
    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required to read PDF files: `pip install pdfplumber`"
            )

        structured_pdf = self.extract_chapters(file)
        doc_title = self.extract_title(structured_pdf)
        header_stack = [doc_title]
        pdf_docs = []
        prev_index = None

        chapters = structured_pdf['chapters']
        for chapter in chapters:
            index = chapter['title'].split(' ')[0]
            index_list = index.split('.')
            text = " ".join([line['title']
                            for line in chapter['subchapters']]).strip()

            # If the index is numeric, update header stack
            if all([item.isnumeric() for item in index_list]):
                title = " ".join(chapter['title'].split(' ')[1:]).strip()

                header_stack_len = len(header_stack) - 1
                index_len = len(index_list)

                # If new subheader present add it
                if header_stack_len < index_len:
                    header_stack.append(title)
                # If going back in hierarchy remove elements to be shorter than index
                elif header_stack_len > index_len:
                    [header_stack.pop()
                     for _ in range(header_stack_len - index_len + 1)]
                    header_stack.append(title)
                else:
                    header_stack.pop()
                    header_stack.append(title)

                # Add document to list
                pdf_docs.append(
                    Document(
                        text=text,
                        metadata={
                            'Filename': "filename",
                            "Content Type": "text",
                            'header_stack': "/".join(header_stack)
                        }
                    )
                )

            # Else just link to parent element
            else:
                pdf_docs.append(
                    Document(
                        text=text,
                        metadata={
                            'Filename': "filename",
                            "Content Type": "text",
                            'header_stack': "/".join(header_stack) + f"/{chapter['title']}"
                        }
                    )
                )
        return pdf_docs

        # with open(file, "rb") as fp:

        #     # Create a PDF object
        #     pdf = pypdf.PdfReader(fp)

        #     # Get the number of pages in the PDF document
        #     num_pages = len(pdf.pages)

        #     # Iterate over every page
        #     text = " ".join([pdf.pages[page].extract_text() for page in range(num_pages)])

        #     metadata = {"file_name": file.name}

        #     if extra_info is not None:
        #         metadata.update(extra_info)

        #     docs = [Document(text=text, metadata=metadata)]
        #     return docs

    @staticmethod
    def extract_title(structured_pdf):
        title = [chapter for chapter in structured_pdf['chapters']
                 if chapter['title'] == 'ÖPUL 2023'][0]['subchapters'][0]['title']

        return title

    @staticmethod
    def extract_chapters(pdf_path):
        """Extracts chapters and subchapters from a pdf file.

        Args:
            pdf_path (String): Path to the pdf file.
        """
        data = {'chapters': []}
        current_chapter = None
        current_subchapter = None

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                lines = text.split('\n')

                for j, line in enumerate(lines):
                    if line.isupper():  # Identifying chapters
                        current_chapter = line.strip()
                        data['chapters'].append(
                            {'title': current_chapter, 'subchapters': []})
                        current_subchapter = None  # Reset subchapter when a new chapter is found
                    elif current_chapter and line.strip() and not line.isnumeric():  # Identifying subchapters
                        current_subchapter = line.strip()
                        data['chapters'][-1]['subchapters'].append(
                            {'title': current_subchapter, 'text': []})
                    elif current_chapter and current_subchapter and line.strip():  # Adding text to subchapter
                        data['chapters'][-1]['subchapters'][-1]['text'].append(
                            line.strip())

        return data
