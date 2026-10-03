# System Architecture

## Objective

Build an end-to-end multimodal RAG pipeline for document summarization and information extraction across PDF text, tables and images.

## Pipeline

```text
PDF
 |
 v
Unstructured partition_pdf
 |
 +------------------+------------------+
 |                  |                  |
Text              Tables            Images
 |                  |                  |
Groq summary      Groq summary      Gemini vision summary
 |                  |                  |
 +------------------+------------------+
                    |
                    v
          all-MiniLM-L6-v2 embeddings
                    |
                    v
                 ChromaDB
                    |
                    v
          MultiVectorRetriever
                    |
                User Query
                    |
                    v
         Retrieved parent content
                    |
          +---------+---------+
          |                   |
       Text/Table           Images
          |                   |
          +---------+---------+
                    |
                    v
          Gemini multimodal reasoning
                    |
                    v
               Final answer
```

## Retrieval design

The vector store indexes compact summaries. Each summary carries a parent `doc_id`. The `MultiVectorRetriever` uses the matching parent IDs to return the original content, preserving richer context for final answer generation.

## Modality handling

- **Text:** extracted from PDF elements and summarized for retrieval.
- **Tables:** preserved as HTML/text when available and summarized separately.
- **Images:** extracted to local files, summarized with a vision model, and retained for multimodal answer generation.

## Engineering notes

The academic report described Groq-hosted LLaMA 4 and Gemini Pro Vision. This implementation keeps the same architectural roles while allowing the model IDs to be configured through `.env`, so supported current endpoints can be selected without changing the application code.
