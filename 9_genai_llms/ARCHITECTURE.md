# Generative AI and LLMs Workshop - Architecture

This module is organized around three architectures that build on each other: a local teaching transformer, a retrieval-grounded chatbot, and an MCP tool server for agent workflows.

## High-Level Architecture

```text
Databricks workspace

  environment/prepare_environment_llm.ipynb
    -> install shared GenAI dependencies
    -> create ai_ml_in_practice.genai_workshop

  Module 1: Tiny Transformer
    9.1_tiny_transformer.ipynb
      -> read workshop text
      -> tokenize
      -> train decoder-only transformer
      -> log metrics and model to MLflow
      -> inspect generation and attention

  Module 2: RAG
    9.2_prepare_knowledge_base.ipynb
      -> parse markdown and notebook text
      -> chunk text with metadata
      -> write Delta tables

    9.3_rag_chatbot.ipynb
      -> create or reuse AI Search endpoint
      -> create or refresh Delta Sync index
      -> retrieve top-k chunks
      -> assemble grounded prompt
      -> call Foundation Model API

  Module 3: Agent MCP
    Databricks App: mcp-genai-workshop
      -> host MCP server over streamable HTTP
      -> expose module metadata tools
      -> expose AI Search retrieval tool

    9.4_agent_mcp_client.ipynb
      -> authenticate to the app
      -> list MCP tools
      -> call selected tools
      -> use an LLM planner to select one MCP tool
```

## Data Plane

```text
Workshop files
  -> documents and notebook markdown
  -> document chunks
  -> Delta table
  -> AI Search Delta Sync index
  -> retrieved context
  -> prompt or MCP tool result
```

Primary table:

```text
ai_ml_in_practice.genai_workshop.document_chunks
```

Important columns:

```text
chunk_id
module_id
module_title
source_path
section_title
chunk_text
retrieval_text
updated_at
```

AI Search index:

```text
ai_ml_in_practice.genai_workshop.document_chunks_rag_index
```

The Delta table is the governed source of truth. The AI Search index is derived serving state optimized for semantic retrieval.

## Module 1: Tiny Transformer

Purpose:

- make tokenization, embeddings, causal attention, and next-token prediction visible
- show a decoder-only transformer in readable TensorFlow/Keras code
- log parameters, per-epoch metrics, final metrics, and a registered model in MLflow
- demonstrate why a tiny model can explain architecture without behaving like a production LLM

Architecture:

```text
workshop corpus
  -> word-level vocabulary
  -> fixed-length token windows
  -> decoder blocks with causal self-attention
  -> next-token prediction
  -> MLflow run and registered model
```

The model is intentionally small and inspectable. It is not a replacement for a pretrained foundation model.

## Module 2: RAG

Purpose:

- show how private workshop knowledge enters an LLM application at runtime
- separate the knowledge base from the model weights
- retrieve relevant context before generation
- cite source paths so answers can be inspected

Architecture:

```text
User question
  -> AI Search similarity search
  -> retrieved chunks with metadata
  -> prompt assembly
  -> Foundation Model API
  -> answer with citations
```

Prompt contract:

```text
System:
  Answer only from the provided context.
  If context is insufficient, say so.
  Cite source_path values.

Context:
  [1] source_path=... section=...
      chunk text...

User:
  question...
```

RAG changes the architecture from model-centric to context-centric. The quality of chunks, metadata, indexing, and retrieval is as important as the LLM endpoint.

## Module 3: Agent MCP

Purpose:

- expose workshop capabilities as tools instead of prompt text
- make tool discovery and invocation explicit through MCP
- host the tool server as a Databricks App
- reuse the RAG index from module 2 through a controlled server boundary

Runtime architecture:

```text
Notebook or agent client
  -> Databricks App URL
  -> MCP streamable HTTP endpoint /mcp
  -> tool schema discovery
  -> tool call
  -> Databricks App service principal
  -> AI Search or static module metadata
  -> structured tool result
```

MCP tools:

```text
health()
  -> returns app status and configured retrieval resources

list_workshop_modules()
  -> returns module ids, titles, and short goals

get_module_summary(module_id)
  -> returns one module summary

search_workshop_docs(query, k)
  -> queries AI Search and returns chunks with citation metadata
```

Authentication and authorization:

- The notebook client exchanges its notebook token for an app-scoped OAuth token before calling the app URL.
- The Databricks App runs as its app service principal.
- The app service principal needs access to the AI Search endpoint, AI Search index, and backing Delta table.
- User-facing app access is controlled through Databricks Apps permissions.

## Production Extensions

The same architecture can be extended with:

- MLflow tracing for prompt, retrieval, model, and tool calls
- curated offline RAG evaluation questions
- improved document parsing and chunking
- reranking before prompt assembly
- ACL-aware retrieval filters
- richer tool schemas for workflow actions
- app CI/CD using Databricks Apps Git deployment or Databricks CLI deployment