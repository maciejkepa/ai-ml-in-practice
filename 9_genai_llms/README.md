# Generative AI and LLMs Workshop

Workshop showing how large language model applications differ from classical ML systems. The module uses three connected demos: a small decoder-only transformer, retrieval-augmented generation, and an MCP tool server hosted as a Databricks App.

## Module Flow

```text
1. Tiny Transformer
   -> train a compact decoder-only model
   -> inspect tokens, causal attention, generation, and MLflow tracking

2. RAG
   -> build a workshop knowledge base
   -> create an AI Search index
   -> generate grounded answers with citations

3. Agent MCP
   -> deploy a Databricks App that exposes MCP tools
   -> call tools from a notebook client
   -> let an LLM select an MCP tool from discovered tool metadata
```

## What You'll Learn

### 1. Tiny Transformer

- How tokens become model inputs
- What causal self-attention does at a high level
- Why decoder-only models predict the next token
- How MLflow tracks model training runs and registered artifacts

### 2. RAG

- How a knowledge base changes an LLM application architecture
- How chunks, metadata, embeddings, and citations work together
- How Databricks AI Search retrieves relevant context
- Why grounded answers depend on retrieved context quality

### 3. Agent MCP

- How tools change an LLM app from passive generation to controlled action
- How MCP standardizes tool discovery and invocation
- How Databricks Apps can host a custom MCP server
- How app identity, permissions, and AI Search access affect tool behavior

## Module Structure

```text
9_genai_llms/
+-- README.md
+-- ARCHITECTURE.md
+-- QUICKSTART.md
+-- src/
|   +-- 9.1_tiny_transformer.ipynb
|   +-- 9.2_prepare_knowledge_base.ipynb
|   +-- 9.3_rag_chatbot.ipynb
|   +-- 9.4_agent_mcp_client.ipynb
|   +-- workshop_decoder.py
+-- app/
    +-- app.yaml
    +-- requirements.txt
    +-- pyproject.toml
    +-- server/
        +-- main.py
```

Shared setup lives outside the module:

```text
environment/
+-- prepare_environment_llm.ipynb
+-- requirements_llm.txt
```

## Databricks Objects

The module uses one catalog/schema namespace:

```text
ai_ml_in_practice.genai_workshop
```

Expected core objects:

```text
document_chunks
document_chunks_rag_index
workshop_decoder_transformer
```

The MCP app uses:

```text
App name: mcp-genai-workshop
AI Search endpoint: genai-workshop-search
MCP endpoint: https://<app-url>/mcp
```

## Key Takeaways

1. **LLM apps are systems**
   - The model is only one component.
   - Retrieval, prompts, permissions, tools, and telemetry shape the final behavior.

2. **RAG is a governed data product**
   - Chunks and metadata are the source of retrieval quality.
   - Citations make answers auditable.
   - Unity Catalog and AI Search provide the control plane and serving layer.

3. **Agents need bounded tools**
   - Tool names, schemas, permissions, and outputs should be narrow and predictable.
   - MCP provides a standard boundary between the agent runtime and external capabilities.

## References

- Databricks AI Search: https://docs.databricks.com/aws/en/ai-search/ai-search
- Databricks Foundation Model APIs: https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/
- Databricks Apps: https://docs.databricks.com/aws/en/dev-tools/databricks-apps/
- Databricks Apps deployment: https://docs.databricks.com/aws/en/dev-tools/databricks-apps/deploy
- Databricks Apps authorization: https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth
- Custom MCP servers on Databricks: https://docs.databricks.com/aws/en/generative-ai/mcp/custom-mcp
