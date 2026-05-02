# Quickstart Guide: MNIST MLOps Workshop

## Overview

This guide walks through the workshop in five modules:

- **Module 1**: Generate the local MLOps stack scaffold
- **Module 2**: Create the serverless base environment from `environment.yml`
- **Module 3**: Train models, deploy the endpoint, and start traffic
- **Module 4**: Query system tables for serving telemetry
- **Module 5**: Trigger and inspect the canary rollout

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Databricks Personal Access Token (PAT)
- Databricks CLI installed: https://docs.databricks.com/en/dev-tools/cli/install
- Workspace admin access for base environment management

## Module 1: Stack Generation

### Step 1: Configure Databricks CLI

```bash
databricks configure --token
```

Or set environment variables:
```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi1234567890abcdefghijklmnopqrst"
```

### Step 2: Clone and navigate

```bash
git clone <repository-url>
cd ai-ml-in-practice/8_mlops
ls -la
```

Expected files:
```text
databricks.yml
environment.yml
resources/
  └── uc_resources.yml
src/
  ├── 8.1_training.ipynb
  ├── 8.2_deployment.ipynb
  ├── 8.3_traffic.ipynb
  └── 8.4_system_tables.ipynb
```

### Step 3: Generate the local scaffold

Run this outside `8_mlops`:

```bash
mkdir -p mlops_stack
cd mlops_stack
databricks bundle init mlops-stacks
```

Expected result:
- a local Declarative Automation Bundle scaffold
- standard project folders such as `resources/` and `src/`

## Module 2: Base Environment Setup

### Step 1: Prepare the environment specification

Use [`environment.yml`](https://github.com/maciejkepa/ai-ml-in-practice/tree/master/8_mlops/environment.yml) as the source specification for the workspace base environment.

Recommended packages:
- `tensorflow`
- `mlflow`
- `databricks-sdk`
- `requests`
- `matplotlib`

### Step 2: Create the workspace base environment

In Databricks:
1. Go to **Settings**
2. Under **Workspace admin**, open **Compute**
3. Next to **Workspace base environments for serverless compute**, click **Manage**
4. Click **Create**
5. Give the environment a name
6. Select the existing `environment.yml` from a Workspace folder or Unity Catalog volume
7. Wait until the environment status is ready

### Step 3: Verify notebook access to the environment

In a serverless notebook:
1. Open the **Environment** side panel
2. Open **Base environment**
3. Select the environment created from `environment.yml`

## Module 3: Training, Deployment, and Traffic

### Step 1: Train and register two CNN models

Run [`src/8.1_training.ipynb`](https://github.com/maciejkepa/ai-ml-in-practice/tree/master/8_mlops/src/8.1_training.ipynb).

Expected outcome:
- two CNN runs in MLflow
- a registered model named `ai_ml_in_practice.mlops_workshop.mnist_cnn`
- `champion` and `challenger` aliases assigned to model versions

### Step 2: Deploy the models to serving endpoint

Run [`src/8.2_deployment.ipynb`](https://github.com/maciejkepa/ai-ml-in-practice/tree/master/8_mlops/src/8.2_deployment.ipynb).

Expected behavior:
- create or update `mnist-serving-endpoint`
- resolve `champion` and `challenger` aliases to concrete Unity Catalog model versions
- route `100%` traffic to `champion`
- update the endpoint to `50/50` traffic between `champion` and `challenger`
- enable usage tracking through the endpoint AI Gateway configuration

### Step 3: Run live traffic simulation notebook

Run [`src/8.3_traffic.ipynb`](https://github.com/maciejkepa/ai-ml-in-practice/tree/master/8_mlops/src/8.3_traffic.ipynb).

Expected behavior:
- send a random MNIST image every 2 seconds
- continue for 10 minutes
- print each endpoint response

## Module 4: System Table Monitoring

### Step 1: Run notebook to read endpoint usage system table

Run [`src/8.4_system_tables.ipynb`](https://github.com/maciejkepa/ai-ml-in-practice/tree/master/8_mlops/src/8.4_system_tables.ipynb).

Look for:
- request volume over time
- requests per minute by served model
- success vs error responses
- which alias-backed version is handling traffic

### Step 2: Verify serving telemetry

Primary source table:

```sql
system.serving.endpoint_usage
```

Expected focus fields:
- `request_time`
- `status_code`
- `requester`
- `served_entity_name`
- `usage_context`

The notebook joins `system.serving.endpoint_usage` with `system.serving.served_entities` on `served_entity_id` so endpoint-level filtering and per-model analytics work correctly.

## Module 5: Canary Rollout

### Step 1: Re-run the canary update

Re-run the second deployment cell in [`8.2_deployment.ipynb`](https://github.com/maciejkepa/ai-ml-in-practice/tree/master/8_mlops/src/8.2_deployment.ipynb).

### Step 2: Observe live traffic

Keep `8.3_traffic.ipynb` running and refresh `8.4_system_tables.ipynb`.

Expected behavior:
- both model aliases receive traffic
- request counts and status mix become comparable across served versions

### Step 3: Decide whether to continue rollout

If the challenger behaves poorly:
- point traffic back to `100% champion`
- keep alias assignments in place until retraining or further analysis

## Optional next steps after successful run

1. Explore **ARCHITECTURE.md** to understand the system design
2. Adjust `environment.yml` to match your workspace package constraints
3. Re-run training with different epoch counts or dataset limits
4. Extend the monitoring notebook with additional SQL aggregations
5. Adapt the alias-routing pattern to a non-MNIST model

## Common commands

```bash
# Validate bundle metadata
databricks bundle validate --target dev

# Deploy bundle resources
databricks bundle deploy --target dev

# View bundle summary
databricks bundle summary --target dev
```
