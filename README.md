Learnings And Projects related to

RETRIEVAL AUGMENTED GENERATION (RAG)
using LangChain and LangGraph


For installation of uv :
curl -LsSf https://astral.sh/uv/install.sh | sh

In terminal:
uv init
uv venv
uv add -r requirements.txt

In Data Ingestion part :
LangChain wants us to convert the given data in particular format :
{
    page_content:"content",
    metadata(dict):""
}

TextLoader and DirectoryLoader 

CharacterTextSplitter - normal,based on given chunk size , no intelligence
RecursiveCharacterTextSplitter - hierarchy wise splitting ["\n\n","\n"] => first \n\n then \n ,intelligent splitting
TokenTextSplitter - based on tokens, it splits

Loading a PDF file
pypdf - simple but slow
pymupdf - speed

Pdf has some issues 
    - may contain images
    - formatting issues

    Creating smart chunks after cleaning those issues and  creating better metadata for chunks 