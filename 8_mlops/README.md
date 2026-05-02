# MNIST MLOps Workshop

Comprehensive workshop demonstrating common **MLOps patterns** for an MNIST canary deployment flow.

## What You'll Learn

This module covers MLOps patterns using Databricks:

### 1. **Declarative Infrastructure as Code**
- Generate an MLOps project scaffold with `databricks bundle init mlops-stacks`
- Define bundle variables and resources in YAML
- Keep infrastructure and application code organized in a standard layout

### 2. **Serverless Environment Management**
- Build a reusable serverless base environment from `environment.yml`
- Reconcile notebook-tested dependencies with workspace-managed environments
- Reuse the same dependency specification across notebooks and jobs

### 3. **Training, Registration, and Alias Management**
- Train two small CNN models on MNIST
- Log both runs to MLflow with Unity Catalog
- Assign `champion` and `challenger` aliases to registered model versions

### 4. **Endpoint Deployment with Alias Routing**
- Create or update a serving endpoint from model aliases
- Resolve `champion` and `challenger` aliases to concrete model versions before deployment
- Start with `100%` traffic on the `champion`
- Shift to `50/50` traffic between `champion` and `challenger`

### 5. **Monitoring with System Tables**
- Query `system.serving.endpoint_usage` and join it with `system.serving.served_entities`
- Review request time, status code, requester, served entity, and usage metadata
- Build per-model traffic and status visualizations

### 6. **Canary Rollout Workflow**
- Keep synthetic traffic flowing while routing changes
- Compare behavior of champion and challenger under live requests
- Use monitored results to decide whether to continue rollout or revert traffic

## Module Structure

```text
8_mlops/
├── README.md                          ← This file (Knowledge & Info)
├── ARCHITECTURE.md                    ← System design & patterns
├── QUICKSTART.md                      ← How to run it
├── databricks.yml                     ← DAB config (Primary)
├── environment.yml                    ← Serverless dependency specification
├── src/                               ← Workshop notebooks
│   ├── 8.1_training.ipynb             ← Train and register two CNN models
│   ├── 8.2_deployment.ipynb           ← Create endpoint and shift traffic
│   ├── 8.3_traffic.ipynb              ← Generate endpoint traffic
│   └── 8.4_system_tables.ipynb        ← Query and visualize serving telemetry
└── resources/                         ← Bundle resources
    └── uc_resources.yml               ← Schema, volume, and experiment definitions
```

# Next steps
- **See QUICKSTART.md** for step-by-step instructions to run the workshop
- **See ARCHITECTURE.md** for detailed design patterns and system architecture
- **Inspect the notebooks** to understand training, deployment, traffic generation, and monitoring
- **Adjust environment dependencies** in `environment.yml` to match your workspace needs
- **Reuse the alias-routing pattern** for your own model-serving workflows

# Key takeaways
1. **Infrastructure as Code (IaC)**
   - Bundle structure, resources, and variables are defined in version-controlled files
   - The project separates configuration from notebook implementation
   - The generated scaffold gives the workshop a repeatable structure
2. **Environment Reproducibility**
   - `environment.yml` becomes the source specification for serverless dependencies
   - Workspace base environments and notebook environments stay aligned
   - Dependency drift is reduced when the exported environment is reused
3. **Alias-Based Deployment**
   - The deployment notebook resolves `champion` and `challenger` aliases to concrete model versions before updating the endpoint
   - Endpoint updates still stay aligned with alias intent without hard-coding versions manually
   - Rollout behavior is easier to reason about during canary testing
4. **Observability Built Into the Platform**
   - System tables provide usage and serving metadata without custom logging code
   - Traffic and status comparisons can be built directly in notebooks
   - Monitoring remains tied to actual served entities and endpoint routing
5. **End-to-End MLOps Flow**
   - The workshop covers scaffold generation, environment setup, model training, serving, traffic, and monitoring
   - Each notebook owns a clear stage in the workflow
   - The structure is small enough for teaching but representative of a real deployment pattern

# For more information:
- Declarative Automation Bundles: https://docs.databricks.com/en/dev-tools/bundles/index.html
- MLflow Documentation: https://mlflow.org/docs/latest/
- Databricks Base Environments: https://learn.microsoft.com/en-us/azure/databricks/admin/workspace-settings/base-environment
