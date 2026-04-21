# Quickstart Guide: MNIST ML Pipeline (Two Phases)

## Overview

This guide walks through deploying and running the MNIST pipeline in two phases:

- **Phase 1** (Development): Train Linear + XGBoost models, test champion comparison logic
- **Phase 2** (Production): Uncomment CNN, enable full 3-model pipeline via CI/CD

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Databricks Personal Access Token (PAT)
- Databricks CLI installed: https://docs.databricks.com/en/dev-tools/cli/install
- GitHub account

## Phase 1: Development Deployment (Linear + XGBoost)

### Step 1: Configure Databricks CLI

```bash
# Authenticate with your Databricks workspace
databricks configure --token

# Enter when prompted:
# - Hostname: https://your-workspace.cloud.databricks.com
# - Token: dapi1234567890... (your PAT)
```

Or set environment variables:
```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi1234567890abcdefghijklmnopqrst"
```

### Step 2: Clone and navigate

```bash
git clone <repository-url>
cd ai-ml-in-practice/7_ml_pipelines
ls -la
```

Expected files:
```
databricks.yml                  ← DAB Bundle config
resources/
  ├── ml_pipeline_job.yml       ← Job DAG
  └── uc_resources.yml          ← Schema + Volume resources
src/
  ├── 7.0_prepare_mnist.ipynb
  ├── 7.1_train_linear.ipynb
  ├── 7.2_train_xgboost.ipynb
  ├── 7.3_train_neural.ipynb     ← Phase 2 only
  ├── 7.4_evaluate_champion.ipynb
  ├── 7.5_evaluate_models.ipynb
  └── 7.6_register_model.ipynb
```

### Step 3: Validate bundle configuration

```bash
# Verify YAML syntax and bundle structure
databricks bundle validate --target dev

# Expected output:
# Validation OK!
```

### Step 4: Deploy to Databricks DEV environment

```bash
# Deploy job, notebooks, and resources to dev workspace
databricks bundle deploy --target dev
```

### Step 5: Run Phase 1 pipeline

#### Option A: Via Databricks UI

1. Go to **Workflows** → **Jobs** → **MNIST ML Pipeline**
2. Click **Run Now**
3. Monitor in the **Runs** tab
4. Expected runtime: **10-15 minutes** (Linear + XGBoost only)

#### Option B: Via CLI

```bash
# Trigger run
databricks bundle run --target dev
```

### Step 6: Verify Phase 1 pipeline execution

Check in Databricks UI:

1. **Jobs & Pipelines** → **MNIST ML Pipeline - Multi-Model Comparison** → **Runs**:
   - prepare_mnist Succeeded
   - train_linear_model Succeeded
   - train_xgboost_model Succeeded
   - evaluate_models Succeeded
   - evaluate_champion Succeeded
   - register_model Succeeded

2. **Experiments** → **/mnist_training**:
   - 2 runs: Linear Regression, XGBoost
   - Metrics: accuracy, f1, auc
   - Models logged with signatures

3. **Catalog** → **ai_ml_in_practice** → **dab_models** → **mnist_digit_classifier**:
   - Version 1: First model registered
   - Alias: `champion` pointing to the latest version

### Step 7: Test champion comparison

Run Phase 1 again:

```bash
# Trigger second run
RUN_ID=$(databricks jobs run-now --job-id $JOB_ID | jq '.run_id')

# Monitor
databricks jobs get-run --run-id $RUN_ID
```

Expected behavior in second run:
- `evaluate_champion` loads current champion (v1)
- Compares new models against champion
- If new best model's F1 > current champion's F1 → Register v2
- Otherwise → Skip registration (no degradation)

## Phase 2: Full pipeline with CI/CD (Linear + XGBoost + CNN)

### Step 1: Prepare repository for CI/CD

The repository already has GitHub Actions workflow configured at `.github/workflows/bundle_ci_cd.yml`

Checkout the `development` branch:
```bash
git checkout development
```

### Step 2: Uncomment CNN task

Edit `resources/ml_pipeline_job.yml`:

Find this section (~line 106):
```yaml
# =============================================
# TASK 4: TRAIN - CNN Neural Network (COMMENTED FOR PHASE 1)
# Uncomment when ready to enable full pipeline in CI/CD
# =============================================
# ...
```

Also update the `evaluate_models` task dependencies to include `train_neural_model`:
```yaml
depends_on:
  - task_key: train_linear_model
  - task_key: train_xgboost_model
  - task_key: train_neural_model  # ← Uncomment this line
```

### Step 3: Commit and push

```bash
git add resources/ml_pipeline_job.yml
git commit -m "Phase 2: Enable CNN in full pipeline"
git push
```

### Step 4: GitHub Actions Auto-Deployment

GitHub Actions workflow (`.github/workflows/bundle_ci_cd.yml`) automatically:

1. Validates DAB configuration
2. Deploys bundle to Databricks dev environment
3. Triggers pipeline run
4. Waits for completion
5. Reports metrics and status

**Check workflow status**:
- Go to repository → **Actions** tab
- Find latest workflow run
- View logs for deployment and execution details

### Step 5: Monitor Phase 2 pipeline

Expected runtime: **30-40 minutes** (3 models, CNN is slow)

```
prepare_mnist
    ↓
    ├─→ train_linear_model (2-3 min)
    ├─→ train_xgboost_model (5-7 min)
    ├─→ train_neural_model (5-7 min)
    ↓
evaluate_models → evaluate_champion → register_model
```

### Step 6: Verify full pipeline results

In Databricks:

1. **Experiments** → **/mnist_training**:
   - 3 runs: Linear, XGBoost, CNN
   - Expected: Linear ~93%, XGBoost ~95%, CNN ~98%

2. **Catalog** → **mnist_digit_classifier**:
   - Version 2+: CNN model as new champion
   - Alias: `champion` pointing to CNN version

## Optional next steps after successful run

1. Explore **ARCHITECTURE.md** to understand design patterns
2. Modify **notebooks** to experiment with hyperparameters
3. Check **MLflow Experiments** to compare model runs
4. View **UC Model Registry** to see versioning and aliasing
5. Extend with your own dataset (replace MNIST)
6. Set up branch protection + required CI/CD checks

## Common commands

```bash
# Validate without deploying
databricks bundle validate --target dev

# Deploy to development
databricks bundle deploy --target dev --force

# Deploy to production (after setup)
databricks bundle deploy --target prod --force

# Run the pipeline
databricks bundle run --target dev telco_ml_pipeline

# View bundle status
databricks bundle summary --target dev

# List jobs in workspace (after deployment)
databricks jobs list --target dev
```