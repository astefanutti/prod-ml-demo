
## 0. First-time setup — credentials

**Do this before any other step.** All Kubernetes secrets in this repo are created from a single `.env` file. The manifests contain `${VAR}` substitution markers — never plain `oc apply` a manifest with credentials directly.

```bash
# 1. Copy the template
cp .env.example .env

# 2. Fill in your values — open .env and set:
#    MINIO_ACCESS_KEY, MINIO_SECRET_KEY
#    REDIS_PASSWORD
#    PG_USER, PG_PASSWORD, PG_DATABASE
#    PG_CLUSTERIP  (get after postgres is deployed: oc get svc postgres -n smartshop -o jsonpath='{.spec.clusterIP}')
#    HF_TOKEN      (from https://huggingface.co/settings/tokens — read scope is enough)
#    MINIO_ENDPOINT_EXTERNAL  (after MinIO route is created)

# 3. Create all Kubernetes secrets at once
make setup-secrets
```

`make setup-secrets` creates these secrets across both namespaces:

| Secret | Namespace | Contains |
|---|---|---|
| `smartshop-credentials` | `smartshop` | MinIO + Redis + Milvus endpoints and keys |
| `redis-credentials` | `smartshop` | `REDIS_PASSWORD` |
| `postgres-credentials` | `smartshop` | `PG_USER`, `PG_PASSWORD`, `PG_DATABASE` |
| `feast-s3-credentials` | `smartshop` | MinIO keys for Feast offline store |
| `feast-redis-secret` | `smartshop` | Redis password for Feast online store |
| `minio-root-user` | `smartshop` | MinIO root credentials |
| `hf-credentials` | `smartshop` | HuggingFace token (key: `token`) |
| `mlflow-s3-credentials` | `redhat-ods-applications` | MinIO keys for MLflow artifact store |
| `mlflow-postgres-secret` | `redhat-ods-applications` | PostgreSQL URI for MLflow backend |

> **Re-run `make setup-secrets` any time you rotate credentials or set up a new cluster.**
> It uses `--dry-run=client -o yaml | oc apply -f -` so it is idempotent.

---

## 1. Verify RHOAI

Go to **Operators → Installed Operators**, switch project to `redhat-ods-operator`, click **Red Hat OpenShift AI**.

Click **Data Science Cluster** tab — confirm `default-dsc` shows **Phase: Ready**.

Click `default-dsc` → scroll to **Conditions** — top-level conditions must all be `True`. `FeastOperatorReady`, `KserveReady`, `MLflowOperatorReady`, `ModelRegistryReady`, and `TrainerReady` must all be `True`.

Or verify via CLI:

```bash
oc get datasciencecluster default-dsc -o jsonpath='{.status.phase}'
# Expected: Ready
```

```bash
oc get datasciencecluster default-dsc -o json \
  | jq '.status.conditions[] | select(.status == "False") | {type, reason}'
```

Any `False` condition other than `SparkOperatorReady`, `LlamaStackOperatorReady`, or `ModelsAsServiceReady` must be resolved before continuing.

---

## 2. Create Namespace

The namespace manifest includes required RHOAI labels:

```bash
oc apply -f infrastructure/smartshop/namespace.yaml
```

This creates the `smartshop` namespace with:
- `app.kubernetes.io/part-of: smartshop-ai`
- `opendatahub.io/dashboard: "true"` — makes it a Data Science Project visible in the RHOAI dashboard

---

## 3. Install Slurm Operator and Deploy Cluster

### 3a — Install the Slinky Operator (OperatorHub)

Go to **Operators → OperatorHub**, search for `slurm`. Select **Slurm Operator** (Community, Red Hat HPC Community). Click the tile to open the detail panel — confirm version `1.0.1-1`, channel `release-1.0`. Click **Install**. Leave all defaults (installs into `slinky` namespace, cluster-wide scope). Click **Install** to confirm. Wait ~1 min until status shows **Succeeded**.

**Verify:**

```bash
oc get pods -n slinky
# slurm-operator-xxxxx         1/1   Running
# slurm-operator-webhook-xxxxx 1/1   Running
```

### 3b — Deploy the Slurm Cluster (Helm)

The operator watches for Slurm CRs. The Helm chart creates them.

```bash
# Namespace with privileged SCC (required by Slurm daemons)
oc adm new-project slurm
oc adm policy add-scc-to-user privileged -n slurm -z default

# Shared home PVC (NFS RWX — accessible from login + all worker pods)
oc apply -f infrastructure/slurm/slurm-home-pvc.yaml

# Deploy Slurm cluster
helm install slurm oci://ghcr.io/slinkyproject/charts/slurm \
  --namespace slurm \
  --version 1.0.1 \
  -f infrastructure/slurm/values.yaml \
  --set-literal "loginsets.slinky.rootSshAuthorizedKeys=$(cat ~/.ssh/id_ed25519.pub)"
```

> **Image version:** `values.yaml` pins to `25.11.1-centos9-ohpc`. Do not change this — earlier builds lack the HTTP health server the operator's liveness probe requires.

**Verify (~3 min for images to pull):**

```bash
oc get pods -n slurm
# slurm-controller-0          3/3   Running
# slurm-login-slinky-xxx      1/1   Running
# slurm-restapi-xxx           1/1   Running
# slurm-worker-slinky-0       2/2   Running
# slurm-worker-slinky-1       2/2   Running

oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo
# PARTITION  AVAIL  TIMELIMIT  NODES  STATE  NODELIST
# slinky        up   infinite      2   idle  slinky-[0-1]
# all*          up   infinite      2   idle  slinky-[0-1]
```

All four CRs (`Controller`, `NodeSet`, `LoginSet`, `RestApi`) should be visible in **Installed Operators → Slurm Operator → All Instances**.

---

## 4. Enable Spark Operator

> **Do not install from OperatorHub/Software Catalog.** The catalog shows several community Spark tiles from `opdev` — avoid them:
> - **Spark Helm Operator** — uses `gcr.io/kubebuilder/kube-rbac-proxy:v0.13.1` which no longer exists; install fails with `ImagePullBackOff`
> - **Spark Application (Operator Backed)** — just a CR template, not an operator
>
> RHOAI 3.4 includes a managed Spark Operator. Enable it via the DataScienceCluster — it's lifecycle-managed and integrates with the DSC health dashboard.

**Via OpenShift Web Console:**

1. Go to **Operators → Installed Operators → Red Hat OpenShift AI**
2. Click **Data Science Cluster → default-dsc → YAML**
3. Find `spec.components.spark.managementState` and set it to `Managed`
4. Click **Save**

**Verify (~2 min):**

```bash
oc get datasciencecluster default-dsc \
  -o jsonpath='{.status.conditions[?(@.type=="SparkOperatorReady")].status}'
# Expected: True

oc get pods -n redhat-ods-applications | grep spark
# spark-operator-controller-xxxxx   1/1   Running
# spark-operator-webhook-xxxxx      1/1   Running
```

**Grant Spark RBAC for the `smartshop` namespace:**

```bash
oc apply -f infrastructure/smartshop/spark-rbac.yaml
```

---

## 5. Deploy MinIO

```bash
oc apply -n smartshop -f - << 'EOF'
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: demo-setup
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: demo-setup-edit
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: edit
subjects:
  - kind: ServiceAccount
    name: demo-setup
---
apiVersion: batch/v1
kind: Job
metadata:
  name: create-s3-storage
spec:
  selector: {}
  template:
    spec:
      containers:
        - args:
            - -ec
            - |-
              oc apply -f https://github.com/rh-aiservices-bu/fraud-detection/raw/main/setup/setup-s3-no-sa.yaml
          command: [/bin/bash]
          image: image-registry.openshift-image-registry.svc:5000/openshift/tools:latest
          name: create-s3-storage
      restartPolicy: Never
      serviceAccountName: demo-setup
EOF
```

Wait for jobs to complete:

```bash
oc get jobs -n smartshop -w
# create-s3-storage        1/1   Complete
# create-minio-root-user   1/1   Complete
# create-minio-buckets     1/1   Complete
```

**Switch to shared NFS storage and set simple credentials:**

```bash
# Scale down
oc scale deployment minio -n smartshop --replicas=0

# Apply the NFS PVC
oc apply -f infrastructure/smartshop/shared-storage.yaml

# Point MinIO at the new PVC
oc patch deployment minio -n smartshop --type=json -p='[
  {"op": "replace", "path": "/spec/template/spec/volumes/0/persistentVolumeClaim/claimName",
   "value": "smartshop-shared-storage"}
]'

# Create minio-root-user secret from .env values
source .env
oc create secret generic minio-root-user \
  --from-literal=MINIO_ROOT_USER="$MINIO_ACCESS_KEY" \
  --from-literal=MINIO_ROOT_PASSWORD="$MINIO_SECRET_KEY" \
  -n smartshop --dry-run=client -o yaml | oc apply -f -

# Scale back up
oc scale deployment minio -n smartshop --replicas=1
oc rollout status deployment/minio -n smartshop

# Delete the old block PVC
oc delete pvc minio -n smartshop
```

**Create SmartShop buckets:**

```bash
source .env
export S3=https://$(oc get route minio-s3 -n smartshop -o jsonpath='{.spec.host}')

for bucket in smartshop-raw smartshop-features smartshop-models smartshop-embeddings milvus; do
  AWS_ACCESS_KEY_ID="$MINIO_ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$MINIO_SECRET_KEY" \
    aws s3 mb s3://$bucket --endpoint-url $S3 --no-verify-ssl
done
```

Update `MINIO_ENDPOINT_EXTERNAL` in `.env` with the MinIO S3 route hostname, then run `make setup-secrets` to create the consolidated `smartshop-credentials` secret:

```bash
# Get the external MinIO S3 URL
oc get route minio-s3 -n smartshop -o jsonpath='{.spec.host}'
# → set MINIO_ENDPOINT_EXTERNAL=https://<that host> in .env

make setup-secrets
```

> **MinIO note:** Open-source MinIO is used here for simplicity. For production use
> **ODF (OpenShift Data Foundation)** or **AIStor** (MinIO's enterprise successor).
> AIStor is available in OperatorHub — search for `minio` in the Software Catalog.

---

## 6. Deploy Redis + RedisInsight UI

```bash
# Secrets must exist before the deployment reads them
make setup-secrets

# Deploy Redis + RedisInsight (envsubst fills ${REDIS_PASSWORD} from .env)
envsubst < infrastructure/redis/redis.yaml | oc apply -f -
oc rollout status deployment/redis deployment/redisinsight -n smartshop
```

This creates:
- `redis-data` PVC — 10Gi, `nfs-csi`, RWX
- `redis-credentials` Secret — password from `REDIS_PASSWORD` in `.env`
- Route for RedisInsight UI

**Verify:**

```bash
source .env
oc exec -it deployment/redis -n smartshop -- \
  redis-cli -a "$REDIS_PASSWORD" ping
# PONG
```

---

## 7. Deploy Milvus + Attu UI

The deploy script handles two known OpenShift issues automatically:

**Issue 1 — etcd SCC:** `milvusdb/etcd` runs as UID 1001, outside OpenShift's default allowed range. The script creates a `milvus` ServiceAccount and grants it `anyuid` SCC.

**Issue 2 — env var injection:** Kubernetes auto-injects `MINIO_PORT`, `MINIO_SERVICE_HOST` etc. from the `minio` Service in the same namespace. These override `values.yaml` and cause Milvus to build a malformed S3 URL (`Endpoint url cannot have fully qualified paths`). The script blanks these injected vars explicitly. ([upstream issue](https://github.com/zilliztech/milvus-helm/issues/99))

```bash
cd infrastructure/milvus && ./deploy.sh smartshop
oc apply -f infrastructure/milvus/attu.yaml
oc rollout status deployment/attu -n smartshop
```

**Verify:**

```bash
oc get pods -n smartshop | grep milvus
# milvus-etcd-0                  1/1   Running
# milvus-standalone-xxxxx        1/1   Running

oc logs -n smartshop deployment/milvus-standalone | grep "ready to serve"
# ---Milvus Proxy successfully initialized and ready to serve!---
```

After MinIO, Redis, and Milvus are up, all four core pods should be running in `smartshop`:

![smartshop namespace pods — milvus-etcd, milvus-standalone, minio, redis all Running](../assets/05-smartshop-pods-running.png)

Once the full demo stack is deployed (including Feast, Grafana, and the Spark History Server), the complete pod list looks like this:

![Full smartshop stack — feast, grafana, milvus, postgres, redis, redsinsight all Running](../assets/openshift-full-stack-pods-running.png)

---

