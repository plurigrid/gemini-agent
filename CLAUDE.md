# gemini-agent — ASI GCP Interleave Stack

Flox environment with Google ADK multi-agent orchestrator.

## Quick Start

```bash
cd ~/i/gemini-agent
flox activate            # installs python3, google-cloud-sdk, uv; creates .venv with google-adk
adk run my_agent         # terminal REPL
adk web my_agent         # browser UI on :8000
```

## Architecture

6 sub-agents under `asi_orchestrator`, mirroring the zig-syrup propagator network:

| Agent | Interleave | Tools |
|-------|-----------|-------|
| VertexAgent | vertex-asi-interleave | vertex_predict, vertex_pipeline_status, gemini_generate |
| BigQueryAgent | bigquery-asi-interleave | bigquery_query, bigquery_table_info |
| ProteinAgent | vertex-ai-protein-interleave | protein_fold_predict, gnomad_variant_query |
| BayesAgent | monad-bayes-asi-interleave | smc/mcmc/pmmh/rmsmc_sample |
| NashatorAgent | game theory solver | nashator_solve, nashator_compose, nashator_gf3_check |
| PropagatorAgent | CellValue merge layer | lattice_merge, trit_*, agm_revise |

## Composite Pipelines

- **GatherAndMerge**: Vertex ‖ BigQuery → PropagatorAgent
- **ProteinPipeline**: ProteinAgent → PropagatorAgent
- **GameVerify**: NashatorAgent → PropagatorAgent (GF(3) check)
- **BayesMerge**: BayesAgent → PropagatorAgent
- **FullInterleave**: ALL 5 data agents in parallel → PropagatorAgent

## Invariants

- CellValue lattice: Nothing < Value < Contradiction (from propagator.zig)
- GF(3) trit conservation across all agent outputs
- AGM belief revision: Levi identity (K - ~p) + p (from continuation.zig)

## Wire Protocol (Nashator)

4-byte big-endian length prefix + JSON-RPC 2.0 over TCP :9999.
Start Nashator: `cd ~/i/nashator && npm start`

## Dependencies

Flox provides: python3, google-cloud-sdk, uv
Pip (via uv): google-adk, google-cloud-aiplatform, google-cloud-bigquery

## GCP Auth

```bash
gcloud auth login
gcloud config set project plurigrid
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com
```
