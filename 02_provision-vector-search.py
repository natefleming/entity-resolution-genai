# Databricks notebook source
# MAGIC %pip install --quiet --upgrade databricks-sdk databricks-vectorsearch mlflow
# MAGIC %restart_python

# COMMAND ----------

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)

vector_search_endpoint_name: str = config.get("vector_search_endpoint_name")
vector_search_index_name: str = config.get("vector_search_index_name")

resolved_entity_table: str = config.get("resolved_entity_table")

primary_key: str = config.get("primary_key")
embedding_source_column: str = config.get("embedding_source_column")

chat_model_name: str =  config.get("chat_model_name")
embedding_model_name: str =  config.get("embedding_model_name")

assert vector_search_endpoint_name is not None
assert vector_search_index_name is not None
assert resolved_entity_table is not None
assert chat_model_name is not None
assert embedding_model_name is not None
assert primary_key is not None
assert embedding_source_column is not None


# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

def endpoint_exists(vsc: VectorSearchClient, vs_endpoint_name: str) -> bool:
    try:
        return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
    except Exception as e:
        if "REQUEST_LIMIT_EXCEEDED" in str(e):
            print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error.")
            return True
        else:
            raise e


vsc: VectorSearchClient = VectorSearchClient()

if not endpoint_exists(vsc, vector_search_endpoint_name):
    vsc.create_endpoint_and_wait(name=vector_search_endpoint_name, verbose=True, endpoint_type="STANDARD")

print(f"Endpoint named {vector_search_endpoint_name} is ready.")


# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.vector_search.index import VectorSearchIndex


def index_exists(vsc: VectorSearchClient, endpoint_name: str, index_full_name: str) -> bool:
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False

if not index_exists(vsc, vector_search_endpoint_name, vector_search_index_name):
  print(f"Creating index {vector_search_index_name} on endpoint {vector_search_endpoint_name}...")
  vsc.create_delta_sync_index_and_wait(
    endpoint_name=vector_search_endpoint_name,
    index_name=vector_search_index_name,
    source_table_name=resolved_entity_table,
    pipeline_type="TRIGGERED",
    primary_key=primary_key,
    embedding_source_column=embedding_source_column, #The column containing our text
    embedding_model_endpoint_name=embedding_model_name #The embedding endpoint used to create the embeddings
  )
else:
  vsc.get_index(vector_search_endpoint_name, vector_search_index_name).sync()

print(f"index {vector_search_index_name} on table {resolved_entity_table} is ready")

# COMMAND ----------



# COMMAND ----------

from typing import Dict, Any, List

import mlflow.deployments
from databricks.vector_search.index import VectorSearchIndex
from mlflow.deployments.databricks import DatabricksDeploymentClient

deploy_client: DatabricksDeploymentClient = mlflow.deployments.get_deploy_client("databricks")

question = "Google"

index: VectorSearchIndex = vsc.get_index(vector_search_endpoint_name, vector_search_index_name)
columns: list[str] = spark.table(resolved_entity_table).columns

search_results: Dict[str, Any] = index.similarity_search(
  query_text=question,
  columns=columns,
  num_results=3)

chunks: List[str] = search_results.get('result', {}).get('data_array', [])
chunks