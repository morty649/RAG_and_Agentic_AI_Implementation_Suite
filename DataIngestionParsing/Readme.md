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

