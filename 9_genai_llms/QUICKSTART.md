# Quickstart Guide: Generative AI and LLMs Workshop

This guide runs the module in three parts:

```text
1. Tiny Transformer
2. RAG
3. Agent MCP
```

## Overview

This guide walks through the workshop in three modules:

- **Module 1**: Train your own Tiny Transformer
- **Module 2**: Create AI Search endpoint for RAG pattern
- **Module 3**: Create your own MCP Agent

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Access to Databricks Foundation Model API pay-per-token endpoints in the workspace region
- Repository connected as a Databricks Git folder
- Permission to create or use Databricks Apps

## Module 1: Tiny Transformer

Run [src/9.1_tiny_transformer.ipynb](https://github.com/maciejkepa/ai-ml-in-practice/tree/master/9_genai_llms/src/9.1_tiny_transformer.ipynb)

Flow:

1. Load workshop markdown files and notebook markdown cells.
2. Build a compact word-level vocabulary.
3. Create fixed-length training windows for next-token prediction.
4. Train a decoder-only transformer with causal self-attention.
5. Log parameters, per-epoch metrics, final metrics, and the TensorFlow model in one MLflow run.
6. Register the model in Unity Catalog.
7. Generate text from short prompts and inspect attention weights.

Expected Unity Catalog model:

```text
ai_ml_in_practice.genai_workshop.workshop_decoder_transformer
```

## Module 2: RAG

RAG uses two notebooks: one prepares the knowledge base, and one creates or reuses the AI Search index and calls the LLM.

### Step 1: Prepare the Knowledge Base

Run [src/9.2_prepare_knowledge_base.ipynb](https://github.com/maciejkepa/ai-ml-in-practice/tree/master/9_genai_llms/src/9.2_prepare_knowledge_base.ipynb)

Default widgets:

```text
catalog_name = ai_ml_in_practice
schema_name = genai_workshop
chunk_size = 700
chunk_overlap = 120
```

Expected tables:

```text
ai_ml_in_practice.genai_workshop.documents
ai_ml_in_practice.genai_workshop.document_chunks
ai_ml_in_practice.genai_workshop.rag_questions
```

Expected chunk columns:

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

### Step 2: Create or Reuse the AI Search Index

Run [src/9.3_rag_chatbot.ipynb](https://github.com/maciejkepa/ai-ml-in-practice/tree/master/9_genai_llms/src/9.3_rag_chatbot.ipynb)

Default retrieval resources:

```text
AI Search endpoint: genai-workshop-search
AI Search index: ai_ml_in_practice.genai_workshop.document_chunks_rag_index
Embedding endpoint: databricks-gte-large-en
LLM endpoint: databricks-meta-llama-3-3-70b-instruct
```

Keep `create_or_update_index=true` when setting up the index. After the index exists and is online, the notebook can be rerun with `create_or_update_index=false`.

Expected behavior:

- create or reuse the AI Search endpoint
- create or refresh the Delta Sync index over `retrieval_text`
- inspect retrieved chunks before calling the LLM
- generate an answer with source citations
- show refusal behavior when retrieved context is insufficient

## Module 3: MCP Server

This module deploys a custom MCP server as a Databricks App and calls it from a notebook client.

The app source folder is:

```text
9_genai_llms/app
```

App files:

```text
app.yaml              # Databricks App command and environment values
requirements.txt      # Python dependencies installed during app deployment
server/main.py         # FastAPI app and MCP tools
```

Configured MCP tools:

```text
health()
list_workshop_modules()
get_module_summary(module_id: int)
search_workshop_docs(query: str, k: int = 3)
```

### Step 1: Confirm RAG Resources Exist

Before deploying the app, run module 2 and confirm these resources exist:

```text
ai_ml_in_practice.genai_workshop.document_chunks
ai_ml_in_practice.genai_workshop.document_chunks_rag_index
genai-workshop-search
```

The MCP search tool uses the same AI Search index as the RAG notebook.

### Step 2: Create the Databricks App

Create a custom Databricks App named:

```text
mcp-genai-workshop
```

The name starts with `mcp-` because Databricks recognizes custom MCP server apps with that naming pattern.

From the Databricks UI:

1. Open **Compute**.
2. Open the **Apps** tab.
3. Click **Create app**.
4. Choose **Custom app**.
5. Use the name `mcp-genai-workshop`.
6. Create the app.

Or create the app with the Databricks CLI:

```bash
databricks apps create mcp-genai-workshop
```

Create the app once. Later source-code changes require redeployment, not a new app.

### Step 3: Choose the App Source Path

The app source must be available to Databricks during deployment.

If the repository is connected as a Databricks Git folder, use the workspace path that ends with:

```text
/ai-ml-in-practice/9_genai_llms/app
```

If the app source is local, upload it to a workspace folder with `databricks sync`.

Using the Databricks CLI from the local app folder:

```bash
cd 9_genai_llms/app
databricks sync . /Workspace/Users/<user-or-folder>/ai-ml-in-practice/9_genai_llms/app
```

For iterative work, keep sync running:

```bash
cd 9_genai_llms/app
databricks sync --watch . /Workspace/Users/<user-or-folder>/ai-ml-in-practice/9_genai_llms/app
```

### Step 4: Deploy the App

Deploy from the Databricks UI:

1. Open **Compute** > **Apps**.
2. Open `mcp-genai-workshop`.
3. Click **Deploy**.
4. Select the source folder:

```text
/Workspace/Users/<user-or-folder>/ai-ml-in-practice/9_genai_llms/app
```

5. Click **Deploy**.
6. Wait until the app status is running.

Or deploy with the Databricks CLI:

```bash
databricks apps deploy mcp-genai-workshop \
  --source-code-path /Workspace/Users/<user-or-folder>/ai-ml-in-practice/9_genai_llms/app
```

The deployment installs `requirements.txt` and starts the command from `app.yaml`:

```text
uvicorn server.main:app --host 0.0.0.0 --port $DATABRICKS_APP_PORT
```

### Step 5: Validate the App

Open the app root URL in the browser. It should return a JSON payload similar to:

```json
{
  "status": "ok",
  "catalog_name": "ai_ml_in_practice",
  "schema_name": "genai_workshop",
  "vector_search_endpoint_name": "genai-workshop-search",
  "vector_search_index_name": "ai_ml_in_practice.genai_workshop.document_chunks_rag_index",
  "databricks_auth_mode": "app_service_principal"
}
```

The MCP endpoint is:

```text
https://<app-url>/mcp
```

When attaching the server in AI Gateway or AI Playground, select the Databricks App named `mcp-genai-workshop`. The public MCP endpoint remains `/mcp`; do not manually append a second `/mcp` if the UI already derives the endpoint from the selected app.

### Step 6: Grant Required Permissions

The app service principal must be able to:

- read `ai_ml_in_practice.genai_workshop.document_chunks`
- query the AI Search endpoint `genai-workshop-search`
- query the AI Search index `ai_ml_in_practice.genai_workshop.document_chunks_rag_index`

Users who call the app also need Databricks App access, for example `CAN USE`.

### Step 7: Call the MCP Server

Run [src/9.4_agent_mcp_client.ipynb](https://github.com/maciejkepa/ai-ml-in-practice/tree/master/9_genai_llms/src/9.4_agent_mcp_client.ipynb)

Set widgets:

```text
mcp_app_name = mcp-genai-workshop
mcp_app_url = https://<app-url>
sample_query = What does the MLOps module deploy?
llm_endpoint_name = databricks-meta-llama-3-3-70b-instruct
```

The notebook:

1. exchanges the notebook token for an app-scoped OAuth token
2. connects to `https://<app-url>/mcp`
3. lists available MCP tools
4. calls module metadata tools
5. calls `search_workshop_docs`
6. asks a Foundation Model endpoint to choose one MCP tool from the discovered tool metadata
7. executes the selected MCP tool and prints the trace


## Troubleshooting

### AI Search Index Is Not Ready

Re-run `src/9.3_rag_chatbot.ipynb` with:

```text
create_or_update_index = true
```

Then inspect the printed index state and wait until the index is online or ready.

### Foundation Model Endpoint Is Unavailable

Switch the `llm_endpoint_name` widget in `src/9.3_rag_chatbot.ipynb` or `src/9.4_agent_mcp_client.ipynb` to an available Foundation Model endpoint from the Serving UI.

### MCP App Is Stopped

Open **Compute** > **Apps**, select `mcp-genai-workshop`, and start or redeploy the app.

### MCP Client Returns 401 Unauthorized

Confirm:

```text
1. mcp_app_name is mcp-genai-workshop.
2. mcp_app_url points to the Databricks App URL.
3. The app is running.
4. The notebook user has access to the app.
5. The notebook configuration cell has been rerun after changing widgets.
```

### MCP Tool Cannot Query AI Search

If `search_workshop_docs()` fails while `health()` works:

```text
1. Open the app root URL and check databricks_auth_mode.
2. Confirm the AI Search endpoint and index names in app.yaml.
3. Grant the app service principal access to the endpoint, index, and backing Delta table.
4. Redeploy the app after changing app code or app.yaml.
```

## References

- Databricks Apps overview: https://docs.databricks.com/aws/en/dev-tools/databricks-apps/
- Deploy Databricks Apps: https://docs.databricks.com/aws/en/dev-tools/databricks-apps/deploy
- Databricks Apps authorization: https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth
- Custom MCP servers: https://docs.databricks.com/aws/en/generative-ai/mcp/custom-mcp