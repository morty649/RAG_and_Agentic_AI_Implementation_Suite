**Overview**

A robust, extensible data ingestion pipeline that converts diverse file formats into LangChain-compatible Document objects (page_content + metadata). Designed to handle real-world messy data (PDFs with images, uneven formatting, complex tables) and create semantically meaningful chunks for downstream retrieval.

Key Features
Multi-format support:

PDFs: Uses pymupdf (fast) with fallback to pypdf; handles images and formatting quirks.

Word documents: Docx2txtLoader for speed, UnstructuredWordDocumentLoader for structure-preserving extraction (titles, headings, tables, page breaks).

CSV/Excel: Custom loaders built on pandas + partition_xlsx (since unstructured had issues) to map columns to page_content and metadata.

JSON: Custom parsing for readable document representation, plus LangChain’s built-in loader as a baseline.

SQL databases: Integration via SQLDatabase utility, with custom loaders to convert query results into document format.

Intelligent chunking strategies:

CharacterTextSplitter for simple splitting.

RecursiveCharacterTextSplitter with hierarchical separators (["\n\n", "\n", " ", ""]) for semantic coherence.

TokenTextSplitter for token-aware splitting (important for LLM context windows).

Smart chunking for PDFs: After cleaning (OCR? image extraction?), creates chunks with rich metadata (page number, source file, section headings) to improve retrieval context.

Metadata enrichment:

Preserves and adds metadata (source, page, creation date, document structure) so that retrieved chunks carry provenance – crucial for grounded generation and citation.

Production-ready design:

Modular functions/classes for each file type, easy to extend.

Handles edge cases: corrupted files, encoding issues, large files.

Configurable chunk sizes and overlap.

Tech Stack
LangChain (loaders, splitters, Document model)

PyMuPDF, pypdf

python-docx, unstructured

SQLite (for databases)

---------------------------------------------------------------------
RAW context - 

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

Parsing a docx
    - Docx2txtLoader => Simple,plain,dumb,fast
    - UnstructuredWordDocumentLoader => Parses the document into semantic elements
        Preserves structure:
        Titles
        Headings
        Paragraphs
        ists
        Tables
        Page breaks
        Adds metadata

Parsing a CSV file and Excel file is quite similar
    - CSVLoader 
    - UnstructuredCSVLoader => typically we create custom meta-data and page_content taken from csv columns

    EXCEL loading is done with pandas and also unstructured but unstructured doesn't seem to work so partition_xlsx did the job

Json parsing is can be done through langchain module or using custom way to get better readable documents

SQL databases can be loaded through 
    
    from langchain_community.utilities import SQLDatabase

or also creating a custom loader which makes it a readable document by LLM
