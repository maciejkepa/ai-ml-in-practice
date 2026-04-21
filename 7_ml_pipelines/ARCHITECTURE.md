# MNIST ML Pipeline - Architecture & Design

Comprehensive overview of the MNIST multi-model classification pipeline architecture, orchestration patterns, and champion comparison workflow.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATABRICKS WORKSPACE                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                   WORKFLOWS / JOBS (Orchestration)                         │ │
│  │                                                                            │ │
│  │                Phase 1: CNN Commented (Initial Development)                │ │
│  │                Phase 2: CNN Enabled (Full Pipeline - CI/CD)                │ │
│  │                                                                            │ │
│  │                      prepare_mnist (Data Prep)                             │ │
│  │                            │                                               │ │
│  │                ┌───────────┼────────────┐                                  │ │
│  │                ▼           ▼            ▼                                  │ │
│  │              ┌────────┬────────────┬────────────┐                          │ │
│  │              │ Linear │ XGBoost    │ CNN        │  (Parallel)              │ │
│  │              │ Model  │ Classifier │ TensorFlow │                          │ │
│  │              └────────┴────────────┴────────────┘                          │ │
│  │                │           │            │                                  │ │
│  │                └───────────┼────────────┘                                  │ │
│  │                            ▼                                               │ │
│  │                     evaluate_models            evaluate_champion           │ │
│  │                  (Compare & Select Best)     (Load Current Champion)       │ │
│  │                            │                            │                  │ │
│  │                            │────────────────────────────┘                  │ │
│  │                            │                                               │ │
│  │                            ▼                                               │ │
│  │                     register_model                                         │ │
│  │               (Register if Better/First Run)                               │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │         MLflow Experiment Tracking (/mnist_training)                       │ │
│  │                                                                            │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │ │
│  │  │ Run: linear  │  │ Run: xgboost │  │ Run: cnn     │                      │ │
│  │  ├──────────────┤  ├──────────────┤  ├──────────────┤                      │ │
│  │  │ Accuracy: 93%│  │ Accuracy: 95%│  │ Accuracy: 98%│ ← Phase 2            │ │
│  │  │ F1: 0.93     │  │ F1: 0.95     │  │ F1: 0.98     │                      │ │
│  │  │ Status: READY│  │ Status: READY│  │ Status: READY│                      │ │
│  │  │ Tags: K/V    │  │ Tags: K/V    │  │ Tags: K/V    │                      │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                      │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                       VOLUME STORAGE                                       │ │
│  │      /Volumes/ai_ml_in_practice/mnist_data/processed/                      │ │
│  │                                                                            │ │
│  │  Training Data:                                                            │ │
│  │  ├─ x_train.npy (47.5 MB) - Flattened 784-d vectors                        │ │
│  │  ├─ y_train.npy (376 KB) - Labels                                          │ │
│  │                                                                            │ │
│  │  Validation Data:                                                          │ │
│  │  ├─ x_val.npy (12 MB) - Flattened 784-d vectors                            │ │
│  │  ├─ y_val.npy (94 KB) - Labels                                             │ │
│  │                                                                            │ │
│  │  Test Data:                                                                │ │
│  │  ├─ x_test.npy (7.8 MB) - Flattened 784-d vectors (Linear, XGBoost)        │ │
│  │  ├─ y_test.npy (78 KB) - Labels                                            │ │
│  │  ├─ x_test_images.npy (31 MB) - 28×28 images (CNN)                         │ │
│  │  └─ y_test_images.npy (78 KB) - Labels                                     │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
                ┌───────────────────┴────────────────────┐
                │                                        │
    ┌───────────────────────┐            ┌──────────────────────┐
    │   Version Control     │            │  CI/CD Automation    │
    │   (GitHub / GitLab)   │            │  (GitHub Actions)    │
    ├───────────────────────┤            ├──────────────────────┤
    │ - Notebooks (Python)  │            │ - Validate DAB YAML  │
    │ - Bundle YAML config  │            │ - Deploy to dev/prod │
    │ - Documentation       │            │ - Trigger pipeline   │
    │ - GitHub Actions      │            │ - Uncomment CNN      │
    │   workflows           │            │ - Report metrics     │
    └───────────────────────┘            └──────────────────────┘
```

## Data Flow & Task Execution

### Phase 1: Linear & XGBoost (Initial Development)

```
Task 0: prepare_mnist
├── Load MNIST from keras.datasets.mnist (60K training + 10K test)
├── Normalize [0, 255] → [0, 1]
├── Split train/val 80/20 on training data
└── Output: 10 numpy arrays saved to volume

Task 1 & 2: train_linear_model, train_xgboost_model (Parallel)
├── Load preprocessed data from volume
├── Train models
├── Evaluate on validation set
├── Log to MLflow with signatures and tags
└── Output: Model run_ids via taskValues

Task 5: evaluate_models
├── Load all trained models from MLflow
├── Evaluate on test set
├── Select best by F1 score
└── Output: best_run_id, best_algorithm, best_f1, best_accuracy

Task 4: evaluate_champion
├── Attempt to load current champion from UC registry
├── If exists: evaluate on test set and capture metrics
├── If not exists: skip (first run)
└── Output: champion_exists, champion_f1, champion_accuracy

Task 6: register_model
├── Compare best model with champion
├── If champion doesn't exist: register best (first run)
├── If best F1 > champion F1: register best as new champion
├── Otherwise: skip registration
└── Output: New model version in UC or no change
```

### Phase 2: With CNN Enabled (Full Pipeline - CI/CD)

Uncomment `train_neural_model` in `ml_pipeline_job.yml`:
- Add Task 3 that trains CNN on original 28×28 images
- Update Task 5 `evaluate_models` depends_on to include `train_neural_model`
- CNN typically achieves highest accuracy (~98%) → likely becomes champion

# Key Patterns

### 1. Champion Comparison Pattern

**Problem**: Prevent model degradation when registering new models.

**Solution**:
- Maintain `champion` alias pointing to current best model
- Before registering: compare new best vs. current champion
- Register only if: first run OR new model is demonstrably better
- Prevents accidental registration of worse models

**Code Pattern**:
```python
# In evaluate_champion task
champion_exists = check_alias_exists(model_name, 'champion')
if champion_exists:
    champion_metrics = load_and_evaluate_champion()
    
# In register_model task
if not champion_exists:
    register_as_champion()  # First run
elif best_f1 > champion_f1:
    register_as_champion()  # Better than current champion
else:
    skip_registration()  # No improvement
```

### 2. Volume Path Configuration

**Pattern**: Centralize data path as DAB variable

**Benefits**:
- Single source of truth for data location
- Easy to change volume without modifying notebooks
- Parameter flows: `databricks.yml` → `ml_pipeline_job.yml` base_parameters → notebook widgets

**Flow**:
```
variables:
  volume_path: "/Volumes/ai_ml_in_practice/mnist_data/processed"
     ↓
base_parameters:
  volume_path: "${var.volume_path}"
     ↓
notebook widget:
  volume_path = dbutils.widgets.get('volume_path')
```

### 3. Model Signatures for UC Registration

**Requirement**: Unity Catalog mandates all models have signatures

**Implementation**:
```python
from mlflow.models import infer_signature

# Create signature from sample data
signature = infer_signature(X_val, y_pred)

# Pass to log_model
mlflow.sklearn.log_model(model, 'model', signature=signature)
```

**Supported Frameworks**:
- Scikit-learn: `infer_signature(X_test, predictions)`
- XGBoost: `infer_signature(X_test, predictions)`
- TensorFlow: `infer_signature(X_test, prediction_proba)`

### 4. Inter-Task Communication via taskValues

**Pattern**: Pass information between tasks in DAB pipeline

**Use Cases**:
- Training tasks → set run_id
- Evaluation tasks → set best model info, champion metrics
- Registration task → uses all upstream info

**Implementation**:
```python
# Set (in upstream task)
dbutils.jobs.taskValues.set(key='best_run_id', value=run_id)

# Get (in downstream task)
best_run_id = dbutils.jobs.taskValues.get(taskKey='evaluate_models', key='best_run_id')
```

### 5. Model Tagging for Governance

**Tag Structure** (Key-Value Pairs):
```python
tags = {
    'environment': 'production',
    'algorithm': 'xgboost',             # Record which algorithm won
    'accuracy': '0.95',                 # Capture metrics
    'f1': '0.95',
    'status': 'champion',               # Mark as current best
    'run_id': 'abcd1234'                # run_id for lineage tracking
}
mlflow.register_model(uri, name, tags=tags)
```

# Key Concepts

## Model Information

### Logistic Regression (Linear)
- **Algorithm**: Scikit-learn LogisticRegression
- **Input**: 784-dimensional flattened images
- **Expected Accuracy**: ~93%
- **Training Time**: ~2 minutes
- **Use Case**: Fast baseline, interpretable
- **Hyperparameters**: max_iter=1000, random_state=15

### XGBoost (Gradient Boosting)
- **Algorithm**: XGBoost classifier
- **Input**: 784-dimensional flattened images
- **Expected Accuracy**: ~95%
- **Training Time**: ~5 minutes
- **Use Case**: Fast, strong performance, good for production
- **Hyperparameters**: n_estimators=50, max_depth=6

### CNN (Convolutional Neural Network) - Phase 2
- **Algorithm**: TensorFlow Keras Sequential
- **Architecture**: Conv2D → MaxPool → Conv2D → MaxPool → Dense → Dropout → Dense(10)
- **Input**: 28×28×1 images (original format)
- **Expected Accuracy**: ~98%
- **Training Time**: ~30 minutes
- **Use Case**: Best accuracy, more training time required
- **Training**: 10 epochs, batch_size=32, EarlyStopping on val_loss

## Data: MNIST Dataset

**Source**: TensorFlow Keras (built-in)

**Size**:
- Training: 60,000 images
- Test: 10,000 images
- Image size: 28×28 pixels (grayscale)
- Classes: 10 (digits 0-9)

**Preprocessing**:
- Normalize pixel values: [0, 255] → [0, 1.0] (divide by 255)
- Flatten: 28×28 → 784 dimensions (for Linear/XGBoost)
- Keep original: 28×28×1 (for CNN)
- Train/val split: 80/20 of training data (48K/12K)

**Storage**: `/Volumes/ai_ml_in_practice/mnist_data/processed/`
- 8 numpy array files saved by `7.0_prepare_mnist.ipynb`
- ~100 MB total size
- Shared across all training notebooks

## Quality Gates

Models must meet both thresholds to pass QA:
- **Minimum Accuracy**: 90%
- **Minimum F1 Score**: 90%

Both Linear and XGBoost typically meet these thresholds.

## Champion Comparison: When to Register

The `register_model` task decides based on:

```
IF best_model.f1 doesn't meet thresholds:
  → Don't register (fail QA)

ELIF no champion exists (first run):
  → Register best_model as champion

ELIF best_model.f1 > champion.f1:
  → Register best_model as new champion

ELSE:
  → Don't register (not better than current champion)
```

## Environment & Dependencies

### Python 3.11+

### Libraries (pinned versions):
- `mlflow==3.6.0` - Experiment tracking & model registry
- `tensorflow==2.20.0` - CNN training
- `xgboost==3.1.3` - Gradient boosting
- `scikit-learn==1.8.0` - Logistic Regression
- `databricks-sdk` - Programmatic Databricks API access
- `numpy` - Array operations
- `pandas` - Data manipulation

### Databricks Requirements:
- Databricks workspace with Unity Catalog enabled
- Personal access token for CLI authentication
- Databricks CLI installed locally

# Modules

## Module 1: Declarative Automation Bundles (DABs) Configuration
Infrastructure is managed as code using `databricks.yml` and `uc_resources.yml`.
- **Environment Targeting**: Native support for `dev` (fast iteration) and `prod` (governed deployment) targets.
- **Resource Provisioning**: Declarative definition of Unity Catalog schemas and Volumes ensures infrastructure is version-controlled and deployed before execution.
- **Variable Injection**: Centralized variables flow from the bundle config into notebook widgets, enabling easy adjustment of thresholds and catalog paths.

### Purpose
Provisioning and managing ML lifecycle infrastructure (schemas, volumes, jobs, and experiments) using a version-controlled, declarative manifest.

### Key Concepts
- **Idempotent Resource Management**: Synchronizing YAML-defined workspace objects with the Databricks REST API for predictable infrastructure state.
- **Variable Interpolation**: Dynamically resolving `${var.*}` placeholders at deployment time to configure compute settings and storage paths.
- **Bundle Lifecycle Management**: Orchestrating build, validation, and deployment phases via the Databricks CLI to ensure configuration integrity.

## Module 2: ML Logic & Experiment Tracking
The core data science logic is contained within the `src/` notebooks, focusing on the MNIST classification task.
- **Experiment Tracking**: Integrated MLflow logging for metrics (Accuracy, F1), parameters, and model signatures for all three algorithms.
- **Storage Strategy**: Uses Unity Catalog Volumes for shared storage of preprocessed numpy arrays, decoupling data from code.
- **Model Governance**: Enforces mandatory model signatures, enabling deployment to Unity Catalog with full lineage.

### Purpose
Executing data processing and model training logic while systematically capturing performance metrics and artifacts in a centralized registry.

### Key Concepts
- **MLflow Entity Persistence**: Programmatically logging metrics (Accuracy, F1), hyperparameters, and dependencies using the MLflow Tracking API.
- **Volume Storage I/O**: Accessing binary datasets (.npy files) via POSIX-compatible paths on managed `/Volumes` for high-throughput training.
- **Model Signature Inference**: Defining strict data contracts (tensor shapes and types) during logging to ensure Model Registry validation.

## Module 3: DAG Orchestration
The multi-task workflow is orchestrated via `ml_pipeline_job.yml`, utilizing advanced Databricks Job patterns.
- **Parallel Task Execution**: Simultaneously trains Linear, XGBoost, and CNN models to maximize cluster utilization.
- **Task Communication**: Uses `dbutils.jobs.taskValues` to pass run metadata and performance metrics between decoupled notebooks.
- **Champion Pattern**: Implements a conditional "Champion vs. Challenger" logic to prevent model degradation in production.

### Purpose
Coordinating the sequence and dependencies of ML tasks to enable parallel experimentation and automated model selection.

### Key Concepts
- **Directed Acyclic Graph (DAG) Execution**: Controlling task execution order and concurrency via `depends_on` definitions in the Job YAML.
- **State Propagation**: Leveraging the `taskValues` API to persist and retrieve run metadata (e.g., `run_id`) across decoupled compute clusters.
- **Registry Metric Comparison**: Programmatically evaluating challenger model performance against the existing version aliased as 'champion'.

## Module 4: CI/CD Automation
Automated life-cycle management is handled by GitHub Actions in `.github/workflows/bundle_ci_cd.yml`.
- **Continuous Integration**: Triggers on Pull Requests to validate bundle syntax, DAG integrity, and variable completeness.
- **Continuous Deployment**: Automatically deploys the validated bundle to the Databricks workspace upon merge to the main branch.
- **Quality Gates**: Integrates `databricks bundle validate` to ensure only high-quality, valid configurations reach the target environment.

### Purpose
Automating the validation and deployment of code and configuration changes to ensure high reliability and continuous delivery.

### Key Concepts
- **Automated Bundle Validation**: Syntactic and structural verification of the project manifest using `databricks bundle validate` within a containerized environment.
- **Headless Authentication Protocol**: Executing secure, non-interactive CLI commands using host and token secrets stored in the CI provider.
- **Event-Driven Workflow Logic**: Implementing path-based filtering and branch-specific triggers to isolate validation (CI) from deployment (CD).