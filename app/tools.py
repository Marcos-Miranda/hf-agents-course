import base64
import mimetypes
import re
from typing import Annotated

import aiofiles  # type: ignore
import nltk  # type: ignore
import whisper  # type: ignore
import wikipedia  # type: ignore
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.tools.base import ArgsSchema
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from app.prompts import DESCRIBE_IMAGE_USER_MESSAGE

nltk.download("punkt_tab")
whisper_model = whisper.load_model("turbo")


def _process_section_content(content: str) -> str:
    text = BeautifulSoup("".join(content), "html.parser").text
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def split_wikipedia_page(page: wikipedia.WikipediaPage, soup: BeautifulSoup) -> list[Document]:
    for edit in soup.find_all(class_="mw-editsection"):
        edit.decompose()
    for tag in soup.find_all(["sup", "figure"]):
        tag.decompose()
    sections = {}
    sections["Summary"] = page.summary.strip()
    current_section = ""
    for elem in getattr(soup.find("div", class_="mw-parser-output"), "children", []):
        if not getattr(elem, "name", None):
            continue
        if elem.name == "div" and "mw-heading2" in elem.get("class", []):
            section_name = elem.text.strip()
            if section_name.lower().replace(" ", "_") in ["further_reading", "external_links", "references"]:
                break
            current_section = section_name
            sections[current_section] = ""
        elif current_section:
            sections[current_section] += str(elem)
    return [
        Document(metadata={"page": page.title, "section": sec_name}, page_content=_process_section_content(sec_content))
        for sec_name, sec_content in sections.items()
    ]


def retrieve_relevant_chunks(
    documents: list[Document],
    query: str,
    k: int = 3,
    text_splitter: TextSplitter | None = None,
    add_metadata: bool = True,
) -> str:
    if text_splitter:
        documents = text_splitter.split_documents(documents)
    if add_metadata:
        documents = [
            Document(
                page_content=f"Metadata: {' '.join(str(item) for item in doc.metadata.items())}\n{doc.page_content}",
                metadata=doc.metadata,
            )
            for doc in documents
        ]
    bm25 = BM25Retriever.from_documents(
        documents=documents, k=k, preprocess_func=lambda x: nltk.word_tokenize(x.lower())
    )
    top_k = bm25.invoke(query.lower())
    return "Relevant chunks:\n\n" + "\n\n".join(doc.page_content for doc in top_k)


@tool
def transcribe_audio(audio_file: Annotated[str, "Path to the audio file"]) -> str:
    """Transcribe an audio file to text."""
    return whisper_model.transcribe(audio_file)["text"].strip()


@tool
def get_wikipedia_page_content(
    page_search_keywords: Annotated[str, "Keywords related to the Wikipedia page to search for"],
    relevant_section_keywords: Annotated[str, "Keywords related to relevant sections of the Wikipedia page"],
) -> str:
    """Download the content of a Wikipedia page related to search keywords and retrieve relevant sections from it.

    For example, if `page_search_keywords` is "Python programming language" and `relevant_section_keywords` is
    "syntax libraries", it will search for a Wikipedia page about Python, download its content, and return sections that
    are relevant to the keywords "syntax" and "libraries".

    When defining `relevant_section_keywords` try not to use too many keywords or too specific keywords, think of it
    as a section or subsection title that you would like to retrieve from the Wikipedia page.
    """
    try:
        page_name = wikipedia.search(page_search_keywords)[0]
    except IndexError as e:
        raise ValueError("No Wikipedia page found for the given search keywords.") from e
    page = wikipedia.page(page_name)
    soup = BeautifulSoup(page.html(), "html.parser")
    documents = split_wikipedia_page(page, soup)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
    return retrieve_relevant_chunks(documents, relevant_section_keywords, k=2, text_splitter=text_splitter)


@tool
def get_website_content(
    url: Annotated[str, "URL of the website to scrape"],
    relevant_content_keywords: Annotated[str, "Keywords related to relevant content on the website"],
) -> str:
    """Download the content of a website and retrieve relevant sections from it.

    For example, if `url` is "https://example.com" and `relevant_content_keywords` is "introduction", it will
    download the content of the website and return sections that are relevant to the keyword "introduction".
    """
    try:
        loader = WebBaseLoader(web_path=url, raise_for_status=True)
        documents = loader.load()
    except Exception as e:
        raise ValueError(f"Failed to retrieve website content: {e}") from e
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return retrieve_relevant_chunks(
        documents, relevant_content_keywords, k=3, text_splitter=text_splitter, add_metadata=False
    )


class DescribeImageInput(BaseModel):
    """Input model for the describe_image tool."""

    image_path: str = Field(description="Path to the image file to describe.")


class DescribeImageTool(BaseTool):
    """Tool to describe an image using Google Generative AI."""

    name: str = "describe_image"
    description: str = "Describes the content of an image."
    args_schema: ArgsSchema | None = DescribeImageInput
    model_name: str = "gemini-2.5-flash-preview-04-17"
    prompt_message: str = DESCRIBE_IMAGE_USER_MESSAGE

    _chat_model: ChatGoogleGenerativeAI = PrivateAttr()

    @model_validator(mode="after")
    def initialize_model(self) -> "DescribeImageTool":
        self._chat_model = ChatGoogleGenerativeAI(model=self.model_name)
        return self

    def _create_message(self, image_bytes: bytes, image_path: str) -> HumanMessage:
        mime_type = mimetypes.guess_type(image_path)[0]
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for {image_path}")
        image_content = base64.b64encode(image_bytes).decode("utf-8")
        return HumanMessage(
            content=[
                {"type": "text", "text": self.prompt_message},
                {
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": mime_type,
                    "data": image_content,
                },
            ]
        )

    def _run(self, image_path: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        """Run the tool with the provided image path."""
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        message = self._create_message(image_bytes=image_bytes, image_path=image_path)
        return str(self._chat_model.invoke([message]).content)

    async def _arun(
        self,
        image_path: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """Run the tool asynchronously with the provided image path."""
        async with aiofiles.open(image_path, "rb") as f:
            image_bytes = await f.read()
        message = self._create_message(image_bytes=image_bytes, image_path=image_path)
        result = await self._chat_model.ainvoke([message])
        return str(result.content)


python_repl_tool = PythonREPLTool(
    description=(
        "A Python shell. Use this to execute python commands. Input should be a valid python command. "
        "Anytime you want to see the output of a value, you should print it out with `print(...)`. "
        "If you get an empty output, "
        "it means that the command executed successfully but there are no `print` statements, so you should add one. "
        "You can use pandas to manipulate structured data. For example, the following command would return the string "
        "representation of a DataFrame: `import pandas as pd\nprint(pd.read_excel('file.xlsx'))`. "
        "You can also open files using the `open` function, e.g. `with open('file.txt', 'r') as f: print(f.read())`."
    )
)
