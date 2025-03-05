# Databricks notebook source
# MAGIC %pip install --upgrade --quiet langchain databricks-langchain databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

catalog_name: str = "nfleming"
database_name: str = "sgws"
volume_name: str = "entity_resolution"

chat_model_name: str = "databricks-meta-llama-3-3-70b-instruct"
embedding_model_name: str = "databricks-gte-large-en"

# COMMAND ----------

from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
  CatalogInfo, 
  SchemaInfo, 
  VolumeInfo, 
  VolumeType,
  SecurableType,
  PermissionsChange,
  Privilege
)


def _volume_as_path(self: VolumeInfo) -> Path:
  return Path(f"/Volumes/{self.catalog_name}/{self.schema_name}/{self.name}")

# monkey patch
VolumeInfo.as_path = _volume_as_path


w: WorkspaceClient = WorkspaceClient()

catalog: CatalogInfo 
try:
  catalog = w.catalogs.get(catalog_name)
except Exception as e: 
  catalog = w.catalogs.create(catalog_name)

schema: SchemaInfo
try:
  schema = w.schemas.get(f"{catalog.full_name}.{database_name}")
except Exception as e:
  schema = w.schemas.create(database_name, catalog.full_name)
  
volume: VolumeInfo
try:
  volume = w.volumes.read(f"{catalog.full_name}.{database_name}.{volume_name}")
except Exception as e:
  volume = w.volumes.create(catalog.full_name, schema.name, volume_name, VolumeType.MANAGED)


spark.sql(f"USE {schema.full_name}")

# COMMAND ----------

import os
import pandas as pd

from langchain_core.documents import Document
from langchain_community.document_loaders import DataFrameLoader


unresolved_path: Path = volume.as_path() / "dataset1.csv"
resolved_path: Path = volume.as_path() / "dataset2.csv"

unresolved_pdf: pd.DataFrame = pd.read_csv(unresolved_path)
resolved_pdf: pd.DataFrame = pd.read_csv(resolved_path)

unresolved_pdf["combined"] = unresolved_pdf.apply(lambda row: " | ".join(row.values), axis=1)
resolved_pdf["combined"] = resolved_pdf.apply(lambda row: " | ".join(row.values), axis=1)

unresolved_loader = DataFrameLoader(unresolved_pdf, page_content_column="combined")
unresolved_documents: list[Document] = unresolved_loader.load()

resolved_loader = DataFrameLoader(resolved_pdf, page_content_column="combined")
resolved_documents: list[Document] = resolved_loader.load()


# COMMAND ----------

resolved_documents

# COMMAND ----------

unresolved_documents

# COMMAND ----------

from databricks_langchain import ChatDatabricks, DatabricksEmbeddings

embeddings = DatabricksEmbeddings(endpoint=embedding_model_name)
llm = ChatDatabricks(endpoint=chat_model_name)

# COMMAND ----------

from langchain_core.vectorstores import VectorStore, InMemoryVectorStore


vector_store = InMemoryVectorStore.from_documents(
    resolved_documents,
    embedding=embeddings,
)

# COMMAND ----------



from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from databricks_langchain import ChatDatabricks, DatabricksEmbeddings

from pydantic import BaseModel, Field



def find_candidate_matches(unresolved_document: Document, vector_store: VectorStore, k: int = 3) -> list[(Document, float)]:
    search_results: list[(Document, float)] = vector_store.similarity_search_with_score(unresolved_document.page_content, k)
    return search_results


class MatchResult(BaseModel):
  document_id: str = Field(..., description="The id of the document")
  similarity_score: float = Field(..., description="The semantic similarity returned from the vector store")
  confidence_score: float = Field(..., description="Confidence score between 0 and 1 (1 being a perfect match)")
  reason: str = Field(..., description="The reason why it was matched or not match")


def verify_match(unresolved_document: Document, resolved_document: Document, similarity_score: float) -> MatchResult:
    prompt = f"""
    You are an english speaking expert in entity resolution. Compare the following two entities based on all available attributes. 
    The simlarity score represents semantic similarity as returned from the vector store.
    You MUST only use the context provided. Do not use any other information.
  
    Document Id: {resolved_document.id}
    Similarity Score: {similarity_score}

    Entity 1: {unresolved_document.metadata}
    Entity 2: {resolved_document.metadata}

    Analyze the similarity of the two entities based on the available data.
    """

    llm_with_tools = llm.with_structured_output(MatchResult)
    match_result: MatchResult = llm_with_tools.invoke(prompt)
 
    return match_result



# COMMAND ----------


def resolve_document(unresolved_document: Document) -> dict[str, ...]:
  match_candidates: list[(Document, float)] = find_candidate_matches(unresolved_document, vector_store)
  match_results: list[MatchResult] = [verify_match(unresolved_document, match_candidate[0], match_candidate[1]) for match_candidate in match_candidates]
  best_match_result: MatchResult = max(match_results, key=lambda x: x.confidence_score)
  matched_document: Document = next(iter([d[0] for d in match_candidates if d[0].id == best_match_result.document_id] or []), None)

  result = {
      "from": unresolved_document.page_content,
      "to": matched_document.page_content if matched_document else None,
      "similarity_score": best_match_result.similarity_score,
      "confidence_score": best_match_result.confidence_score,
      "reason": best_match_result.reason,
  }

  print(result)
  
  return result




# COMMAND ----------


matches: list[dict[str, ...]] = [resolve_document(d) for d in unresolved_documents]

# Convert results into DataFrame
matched_df = pd.DataFrame(matches)

# Save results
#matched_df.to_csv("entity_resolution_results.csv", index=False)

# Print sample output
display(matched_df)