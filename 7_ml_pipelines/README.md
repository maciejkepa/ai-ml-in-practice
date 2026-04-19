# MNIST ML Pipeline - Multi-Model Classification

Comprehensive pipeline demonstrating **Databricks Declarative Automation Bundles (DABs)**, **MLflow tracking**, **model versioning with Unity Catalog**, and **champion comparison patterns** for MNIST digit classification (0-9).

## What You'll Learn

This module covers ML engineering patterns using Databricks:

### 1. **Declarative Infrastructure as Code (DABs)**
- Define resources (schemas, volumes, jobs) in YAML
- Deploy infrastructure and applications together
- Auto-generate Terraform configurations
- Environment-specific deployment (dev vs. prod)

### 2. **Multi-Model Training & Selection**
- Train multiple algorithms in parallel
- Automatic comparison and ranking by metrics
- Select best model based on chosen metric (F1, accuracy, etc.)
- Log all runs to MLflow for full experiment tracking

### 3. **Champion Comparison Pattern**
- Maintain a "champion" alias for the current best model version
- Prevent model degradation by comparing before promotion
- Only register new models if: first deployment OR better metrics
- Full audit trail: why each model was promoted or rejected

### 4. **Model Governance with Unity Catalog**
- Register models in UC (not local MLflow)
- Model signatures required (input/output schema)
- Version control for models (v1, v2, v3, ...)
- Alias-based routing (champion, staging, production)
- Tag-based metadata (algorithm, accuracy, deployment reason)

### 5. **Parameter Pipeline & Configuration**
- Single source of truth: DAB variables in `databricks.yml`
- Parameters flow: Job → base_parameters → Notebook widgets
- Easy to change model registry path, volume location, thresholds without code changes

### 6. **Inter-Task Communication**
- DAB pipeline tasks communicate via `dbutils.jobs.taskValues`
- Upstream tasks set values, downstream tasks retrieve them
- Use cases: pass best model info, champion metrics, deployment decisions

### 7. **CI/CD Integration**
- GitHub Actions workflow auto-deploys on push
- Two phases: Phase 1 (dev with 2 models) → Phase 2 (prod with 3 models)
- Automated validation, deployment, and pipeline execution

## Module Structure

```
7_ml_pipelines/
├── README.md                            ← This file (Knowledge & Info)
├── ARCHITECTURE.md                      ← System design & patterns
├── QUICKSTART.md                        ← How to run it
├── databricks.yml                       ← DAB config (Primary)
├── .gitignore
├── .github/
│   └── workflows/
│       └── bundle_ci_cd.yml             ← GitHub Actions CI/CD
├── src/                                 ← ML Training Notebooks
│   ├── 00_prepare_mnist.ipynb           ← Load MNIST, save to Volume
│   ├── 01_train_linear.ipynb            ← Logistic Regression (always on)
│   ├── 02_train_xgboost.ipynb           ← XGBoost (always on)
│   ├── 03_train_neural.ipynb            ← CNN TensorFlow (commented Phase 1)
│   ├── 04_evaluate_champion.ipynb       ← Load champion from registry
│   ├── 05_evaluate_models.ipynb         ← Compare trained models
│   └── 06_register_model.ipynb          ← Register if better than champion
└── resources/                           ← Orchestration & Resources
    ├── ml_pipeline_job.yml              ← Job DAG & task dependencies
    └── uc_resources.yml                 ← Schema & volume definitions
```

# Next steps
- **See QUICKSTART.md** for step-by-step instructions to run the pipeline
- **See ARCHITECTURE.md** for detailed design patterns and system architecture
- **Explore the notebooks** to understand each training and evaluation step
- **Modify hyperparameters** to experiment with model performance
- **Extend with your own data** to apply patterns to real-world datasets

# Key takeaways
1. **Infrastructure as Code (IaC)**
   - All infrastructure defined in version-controlled files (YAML)
   - Deployment is deterministic and repeatable
   - Easy to restore after infrastructure failure
2. **Separation of Concerns**
   - ML Logic: Python notebooks in /src
   - Orchestration: YAML configuration in /resources
   - Infrastructure: CI/CD pipelines in .github/workflows
3. **Quality Gates**
   - Validation prevents bad models from reaching production
   - Task 2 exit code controls Task 3 execution
   - Automated workflow prevents human error
4. **Full Lineage Tracking**
   - MLflow tracks training parameters and metrics
   - Databricks tracks job execution history
   - Unity Catalog tracks model registration and access
   - GitHub tracks code changes and deployments
5. **Automation & Scale**
   - CI/CD automates validation and deployment
   - No manual steps required after code merge
   - Ready for scheduled retraining pipelines
6. **Production Ready**
   - Error handling and retry logic
   - Timeout configuration
   - Notification system (email on failure)
   - Compatible with Databricks Free Edition (serverless compute)

# For more information:
- Declarative Automation Bundles: https://docs.databricks.com/en/dev-tools/bundles/index.html
- MLflow Documentation: https://mlflow.org/docs/latest/
- GitHub Actions Documentation: https://docs.github.com/en/actions
