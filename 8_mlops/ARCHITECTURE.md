# MNIST MLOps Workshop - Architecture & Design

Comprehensive overview of the MNIST workshop architecture, serving rollout patterns, and monitoring workflow.

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATABRICKS WORKSPACE                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    NOTEBOOK WORKFLOW (Execution)                           │ │
│  │                                                                            │ │
│  │                      8.1_training (Model Registration)                     │ │
│  │                                │                                           │ │
│  │                                ▼                                           │ │
│  │                     8.2_deployment (Endpoint Updates)                      │ │
│  │                           ├─ 100% champion                                 │ │
│  │                           └─ 50/50 champion/challenger                     │ │
│  │                                │                                           │ │
│  │                                ▼                                           │ │
│  │                      8.3_traffic (Live Request Simulation)                 │ │
│  │                                │                                           │ │
│  │                                ▼                                           │ │
│  │                   8.4_system_tables (Telemetry Analysis)                   │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                       MLflow Experiment Tracking                           │ │
│  │                                                                            │ │
│  │       ┌──────────────┐                ┌──────────────┐                     │ │
│  │       │ Run: model_a │                │ Run: model_b │                     │ │
│  │       ├──────────────┤                ├──────────────┤                     │ │
│  │       │ CNN metrics  │                │ CNN metrics  │                     │ │
│  │       │ Signature    │                │ Signature    │                     │ │
│  │       │ UC version   │                │ UC version   │                     │ │
│  │       └──────────────┘                └──────────────┘                     │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    UNITY CATALOG MODEL REGISTRY                            │ │
│  │                                                                            │ │
│  │  ai_ml_in_practice.mlops_workshop.mnist_cnn                                │ │
│  │  ├─ v1, v2, v3, ...                                                        │ │
│  │  ├─ @champion                                                              │ │
│  │  └─ @challenger                                                            │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                     MODEL SERVING + SYSTEM TABLES                          │ │
│  │                                                                            │ │
│  │  Endpoint: mnist-serving-endpoint                                          │ │
│  │  ├─ Served entity: champion                                                │ │
│  │  ├─ Served entity: challenger                                              │ │
│  │  ├─ Traffic split: 100/0 → 50/50                                           │ │
│  │  └─ Monitoring source: system.serving.endpoint_usage                       │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow & Task Execution

### Module 1 & 2: Setup

```text
Module 1: Stack Generation
├── Run databricks bundle init mlops-stacks locally
├── Inspect generated databricks.yml, resources/, and src/
└── Output: Standard local DAB scaffold

Module 2: Base Environment
├── Start from environment.yml
├── Create workspace base environment in Databricks settings
├── Select that environment in serverless notebooks
└── Output: Reusable serverless dependency environment
```

### Module 3, 4, and 5: Training, Serving, and Monitoring

```text
Notebook 8.1: training
├── Load MNIST from tensorflow.keras.datasets
├── Normalize images and shape tensors for CNN input
├── Train two small CNN models
├── Log both runs to MLflow with signatures
├── Register versions in Unity Catalog
└── Output: champion and challenger aliases

Notebook 8.2: deployment
├── Resolve champion and challenger aliases to concrete model versions
├── Create or update mnist-serving-endpoint
├── First config: 100% traffic to champion version
├── Second config: 50/50 traffic split
└── Enable AI Gateway usage tracking

Notebook 8.3: traffic
├── Load MNIST test images
├── Select one random image every 2 seconds
├── Invoke serving endpoint
└── Output: live request stream and printed responses

Notebook 8.4: system tables
├── Join system.serving.endpoint_usage with system.serving.served_entities
├── Filter to mnist-serving-endpoint
├── Inspect request_time, status_code, requester, served_entity_name
└── Output: plots for per-model traffic and status distribution
```

# Key Patterns

### 1. Alias-First Deployment

**Problem**: Avoid embedding hard-coded model versions into serving configs.

**Solution**:
- use `champion` and `challenger` aliases as the source of truth
- resolve those aliases to concrete versions immediately before endpoint updates
- keep rollout intent alias-based while satisfying the serving API contract

**Code Pattern**:
```text
alias -> model version lookup
champion -> vN
challenger -> vM
```

### 2. Shared Environment Specification

**Pattern**: Use `environment.yml` as the source specification for serverless dependencies.

**Benefits**:
- one dependency definition for notebook testing and workspace base environments
- easier package reconciliation across serverless notebooks
- fewer manual package differences between users

**Flow**:
```text
environment.yml
   ↓
Workspace base environment
   ↓
Notebook Environment panel
   ↓
Workshop notebooks
```

### 3. Two-Step Traffic Routing

**Pattern**: Deploy safely by separating initial endpoint creation from canary routing.

**Implementation**:
- first create or update the endpoint with `100%` traffic on `champion`
- then apply a second config with `50/50` traffic
- keep traffic generation running while the route changes

### 4. Monitoring from System Tables

**Pattern**: Use build-in endpoint monitoring.

**Implementation**:
```sql
SELECT se.endpoint_name, se.served_entity_name, eu.request_time, eu.status_code
FROM system.serving.endpoint_usage eu
JOIN system.serving.served_entities se
  ON eu.served_entity_id = se.served_entity_id
WHERE se.endpoint_name = 'mnist-serving-endpoint'
```

**Benefits**:
- endpoint-level filtering through served entity metadata
- per-model traffic and status analysis by served model version
- possible extension with Inference Tables or OpenTelemetry

# Key Concepts

## Model Information

### Model A
- **Algorithm**: TensorFlow Keras CNN
- **Input**: 28×28×1 image tensors
- **Use Case**: first candidate for champion selection
- **Training Goal**: fast notebook-friendly CNN baseline

### Model B
- **Algorithm**: TensorFlow Keras CNN
- **Input**: 28×28×1 image tensors
- **Use Case**: challenger candidate with different architecture
- **Training Goal**: comparable model for alias-based rollout

## Data: MNIST Dataset

**Source**: TensorFlow Keras (built-in)

**Size**:
- Training images from the standard MNIST training split
- Test images from the standard MNIST test split
- Image size: 28×28 grayscale
- Classes: 10 digits

**Preprocessing**:
- normalize pixel values to `[0, 1]`
- add channel dimension for CNN input
- keep a reduced training subset to keep notebook execution practical

## Deployment Behavior

The serving workflow uses:
- one registered model name: `ai_ml_in_practice.mlops_workshop.mnist_cnn`
- two aliases: `champion` and `challenger`
- one endpoint: `mnist-serving-endpoint`
- one AI Gateway usage tracking configuration

The rollout sequence is:
```text
1. Register model versions
2. Assign aliases
3. Resolve aliases to versions
4. Deploy 100% champion
5. Start traffic
6. Update to 50/50
7. Inspect telemetry
```

## Environment & Dependencies

### Serverless base environment
- sourced from `environment.yml`
- created in workspace settings
- selected in the notebook Environment panel

### Core libraries
- `tensorflow`
- `mlflow`
- `databricks-sdk`
- `requests`
- `matplotlib`

# Modules

## Module 1: Declarative Automation Bundles Configuration
Infrastructure starts with a generated DAB scaffold and bundle configuration.
- **Project scaffold**: `databricks bundle init mlops-stacks`
- **Primary config**: `databricks.yml`
- **Resource config**: `resources/uc_resources.yml`

## Module 2: Serverless Base Environment Management
Dependency configuration is centered on `environment.yml`.
- **Environment source**: exported YAML specification
- **Workspace management**: base environment created in Databricks settings
- **Notebook usage**: selected through the Environment side panel

## Module 3: Training & Model Registration
The ML logic is contained in `8.1_training.ipynb`.
- **Experiment tracking**: MLflow metrics, parameters, and signatures
- **Registry target**: `ai_ml_in_practice.mlops_workshop.mnist_cnn`
- **Alias assignment**: `champion` and `challenger`

## Module 4: Serving & Traffic
The serving flow spans notebooks `8.2` and `8.3`.
- **Endpoint management**: create or update `mnist-serving-endpoint`
- **Version resolution**: map aliases to explicit Unity Catalog model versions before deployment
- **Traffic simulation**: send one MNIST request every 2 seconds
- **Rollout behavior**: `100% champion` then `50/50`

## Module 5: Monitoring & Canary Evaluation
Observability is handled in `8.4_system_tables.ipynb`.
- **Source tables**: `system.serving.endpoint_usage` and `system.serving.served_entities`
- **Key fields**: request_time, status_code, requester, served_entity_name, usage_context
- **Decision support**: compare request volume and status mix by served model
