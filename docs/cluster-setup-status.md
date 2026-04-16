# Cluster Setup Status
**Cluster:** `api.oai-kft-ibm.ibm.rh-ods.com`
**RHOAI Version:** 3.4 EA2 RC2
**Checked:** 2026-04-10
**Namespace to create:** `smartshop`

---

## Summary

| Category | Ready | Needs Work | Missing |
|---|:---:|:---:|:---:|
| Kubeflow Trainer v2 | ✅ | | |
| ClusterTrainingRuntimes (CUDA) | ✅ | | |
| KServe | ✅ | | |
| RHOAI Model Registry | ✅ | | |
| Feast Operator | ✅ | | |
| GPU Nodes (2 × 8) | ✅ | | |
| Storage Classes | ✅ | | |
| MLflow | | ⚠️ CrashLoopBackOff | |
| `smartshop` Namespace | | | ❌ |
| Spark Operator | ✅ | | |
| Slurm / Slinky | | | ❌ |
| MinIO (smartshop) | | | ❌ |
| Redis | ✅ | | |
| Milvus | ✅ | | |
| Feast FeatureStore CR | | | ❌ |

---

## ✅ Ready — No Action Needed

### Kubeflow Trainer v2
- CRDs present: `trainjobs.trainer.kubeflow.org`, `trainingruntimes.trainer.kubeflow.org`, `clustertrainingruntimes.trainer.kubeflow.org`
- Installed: 2026-01-09

### ClusterTrainingRuntimes
Available CUDA runtimes — pick one for our TrainJob specs:

| Runtime | CUDA | PyTorch | Notes |
|---|---|---|---|
| `torch-distributed-cuda128-torch29-py312` | 12.8 | 2.9 | Stable |
| `torch-distributed-cuda130-torch291-py312` | 13.0 | 2.9.1 | Latest |
| `torch-distributed` | unknown | unknown | Generic, check image |
| `torch-default-gpu` | unknown | unknown | Check image |

> **Action:** Update `infrastructure/openshift/trainjobs.yaml` — set `runtimeRef.name` to `torch-distributed-cuda130-torch291-py312` (or confirm CUDA version matches node drivers).

### Kueue
- All CRDs present
- Existing ClusterQueues: `training-cluster-queue`, `default`, `staging`, `notebooks-cluster-queue`, `serving-cluster-queue`
- ResourceFlavors: `nvidia-gpu-flavor`, `gpu-flavor`, `demo-gpu-flavor`, `default-flavor`
- > **Action:** Create a `LocalQueue` in `smartshop` namespace pointing to `training-cluster-queue`. Update `infrastructure/slurm/kueue-config.yaml` if needed.

### KServe
- CRD present: `inferenceservices.serving.kserve.io`
- No `ClusterServingRuntimes` found — may be namespace-scoped or using inline runtime specs
- > **Action:** Verify vLLM runtime is available or add `ServingRuntime` for vLLM to `smartshop` namespace.

### RHOAI Model Registry
- Instance `default-modelregistry` is `Ready: True`
- Operator running in `redhat-ods-applications`
- > **Action:** Enable Model Registry access for `smartshop` namespace (add namespace to allowed list in RHOAI Dashboard).

### Feast Operator
- CRD present: `featurestores.feast.dev`
- Operator pod running: `feast-operator-controller-manager` in `redhat-ods-applications`
- Existing FeatureStores in: `astefanu` (banking, example), `nkathole` (banking, salesforecasting)
- > **Action:** Apply `feast/feature_repo/` as a `FeatureStore` CR in `smartshop` namespace.

### GPU Nodes
```
NAME                            GPU   CPU      MEM
oai-kft-ibm-jcsbk-gpu-1-msf4t   8     79500m   ~1.2Ti
oai-kft-ibm-jcsbk-gpu-2-8gmgw   8     79500m   ~1.2Ti
```
- 16 GPUs total
- Sufficient for: DDP rec model (4 GPUs, 1 node) + FSDP LLM (8 GPUs, 2 nodes)
- > **Action:** Check GPU type (`nvidia-smi` or node labels) — confirm A100/H100 for FSDP memory requirements.

### Storage Classes
| Class | Provisioner | Default |
|---|---|---|
| `ibmc-vpc-block-10iops-tier` | `vpc.block.csi.ibm.io` | ✅ Yes |
| `ibmc-vpc-block-5iops-tier` | `vpc.block.csi.ibm.io` | No |
| `nfs-csi` | `nfs.csi.k8s.io` | No |

> Use `nfs-csi` for shared ReadWriteMany volumes (MinIO, Feast offline store). Use `ibmc-vpc-block-10iops-tier` for block storage (Redis, Milvus).

---

## ⚠️ Needs Fix

### MLflow
- Operator pod running: `mlflow-operator-controller-manager` in `redhat-ods-applications`
- OAuth proxy running: `mlflow-oauth-proxy` in `redhat-ods-applications`
- **Main pod `mlflow` is in `CrashLoopBackOff`**
- [ ] Check logs: `oc logs -n redhat-ods-applications mlflow-67b969f8d4-d7q2n`
- [ ] Identify crash cause (likely DB connection or S3 config issue)
- [ ] Fix or request Karel to fix
- [ ] Alternatively: deploy a standalone MLflow instance in `smartshop` namespace

---

## ❌ Missing — Needs Setup

### `smartshop` Namespace
- [ ] `oc new-project smartshop`
- [ ] Add RHOAI `ODHProject` label if required by platform
- [ ] Set resource quotas (coordinate with Karel to avoid conflicting with other teams)

### Spark Operator ✅
- Enabled via `default-dsc` DataScienceCluster — `managementState: Managed` set on 2026-04-13
- Pods running in `redhat-ods-applications`: `spark-operator-controller`, `spark-operator-webhook`
- CRDs registered: `sparkapplications.sparkoperator.k8s.io`, `scheduledsparkapplications.sparkoperator.k8s.io`, `sparkconnects.sparkoperator.k8s.io`
- DSC condition `SparkOperatorReady` initializing — will flip to `True` once 2/2 deployments ready
- [x] Enabled in `default-dsc` DSC spec
- [ ] Verify `SparkOperatorReady: True` in DSC conditions
- [ ] Create `spark` service account + RBAC in `smartshop` namespace
- [ ] Apply SparkApplication manifests from `infrastructure/openshift/`

### Slurm / Slinky Operator
- No Slurm or Slinky pods/CRDs found anywhere
- This is required for the LLM FSDP TrainJob (`smartshop-llm-finetune` uses `ClusterTrainingRuntime` with Slurm)
- **Options:**
  - A) Request Karel install Slinky operator (preferred — required for Slurm demo story)
  - B) Fallback: run FSDP job directly on K8s using `torch-distributed-cuda130-torch291-py312` ClusterTrainingRuntime (no Slurm, weaker demo narrative)
- [ ] Confirm with Karel if Slurm/Slinky can be installed
- [ ] If yes: apply `infrastructure/slurm/kueue-config.yaml` after Slinky is up
- [ ] If no: update `trainjobs.yaml` to use K8s-native runtime for LLM job

### MinIO (smartshop)
- Only existing MinIO PVC found: `efazal-models/minio` (another user's)
- We need our own MinIO for `smartshop` S3 bucket
- [ ] Deploy MinIO in `smartshop` namespace (or request shared S3 endpoint from Karel)
- [ ] Create bucket: `smartshop`
- [ ] Create sub-paths: `raw/`, `features/`, `models/`, `embeddings/`
- [ ] Store credentials in a Secret: `smartshop-s3-credentials`
- [ ] Update all manifests and configs with MinIO endpoint + credentials

### Redis ✅
- Deployed 2026-04-13 in `smartshop` namespace
- Image: `quay.io/opstree/redis:v7.0.5` (Apache 2.0, OpenShift-compatible)
- Storage: `redis-data` PVC, 10Gi, `ibmc-vpc-block-10iops-tier` (RWO)
- Endpoint: `redis.smartshop.svc.cluster.local:6379`
- Credentials: Secret `redis-credentials`, password `smartshop-redis-2026`
- Manifest: `infrastructure/redis/redis.yaml`
- [x] Pod running 1/1
- [ ] Update `feast/feature_repo/feature_store.yaml` with Redis connection string

### Milvus ✅
- Deployed 2026-04-13 in `smartshop` namespace
- Chart: `milvus/milvus` v4.2.58 (Milvus 2.5.16), standalone mode
- Storage: `milvus` PVC 20Gi block (RocksMQ WAL), `data-milvus-etcd-0` 10Gi block (etcd)
- Vector segment data stored in MinIO bucket `milvus` via S3 API (indirectly on NFS)
- Endpoint: `milvus.smartshop.svc.cluster.local:19530` (gRPC), `:9091` (REST)
- Values: `infrastructure/milvus/values.yaml`
- Deploy script: `infrastructure/milvus/deploy.sh`
- Known issues documented in `docs/setup-guide/06-milvus.md`:
  - etcd requires `anyuid` SCC (UID 1001) — SA `milvus` created + bound
  - Kubernetes injects `MINIO_PORT` env var from same-namespace Service, overrides config — blanked post-install
- [x] Both pods running 1/1 (`milvus-standalone`, `milvus-etcd-0`)
- [ ] Update `feast/feature_repo/feature_store.yaml` with Milvus host
- [ ] Update `infrastructure/openshift/inferenceservices.yaml` `MILVUS_HOST` env var

### Feast FeatureStore CR (smartshop)
- Feast operator is ready but no `FeatureStore` CR exists for `smartshop`
- Depends on MinIO, Redis, Milvus being up first
- [ ] Update `feast/feature_repo/feature_store.yaml` with cluster endpoints
- [ ] Apply FeatureStore CR: `oc apply -f feast/feature_repo/feature_store.yaml -n smartshop`
- [ ] Verify: `oc get featurestore -n smartshop`

---