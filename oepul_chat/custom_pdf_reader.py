import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from llama_index import SimpleDirectoryReader
from llama_index.readers.base import BaseReader
from llama_index.schema import Document, NodeRelationship, RelatedNodeInfo
from llama_index.readers.file.docs_reader import DocxReader, HWPReader, PDFReader


logger = logging.getLogger(__name__)


class CustomPDFReader(BaseReader):
    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required to read PDF files: `pip install pypdf`"
            )
        with open(file, "rb") as fp:

            # Create a PDF object
            pdf = pypdf.PdfReader(fp)

            # Get the number of pages in the PDF document
            num_pages = len(pdf.pages)

            # Iterate over every page
            text = " ".join([pdf.pages[page].extract_text() for page in range(num_pages)])
            
            metadata = {"file_name": file.name}

            if extra_info is not None:
                metadata.update(extra_info)

            docs = [Document(text=text, metadata=metadata)]
            return docs
