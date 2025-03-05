# Databricks notebook source
# MAGIC %pip install --upgrade --quiet langchain langchain-openai databricks-langchain databricks-sdk mlflow python-dotenv
# MAGIC %restart_python

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)

unresolved_entity_path: str = config.get("unresolved_entity_path")

vector_search_endpoint_name: str = config.get("vector_search_endpoint_name")
vector_search_index_name: str = config.get("vector_search_index_name")
embedding_source_column: str = config.get("embedding_source_column")
primary_key: str = config.get("primary_key")

chat_model_name: str =  config.get("chat_model_name")
embedding_model_name: str = config.get("embedding_model_name")

resolved_entity_table: str = config.get("resolved_entity_table")


assert vector_search_endpoint_name is not None
assert vector_search_index_name is not None
assert chat_model_name is not None
assert embedding_model_name is not None
assert embedding_source_column is not None
assert unresolved_entity_path is not None
assert resolved_entity_table is not None
assert primary_key is not None

# COMMAND ----------

import os
import re

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from langchain_core.documents import Document
from langchain_community.document_loaders import DataFrameLoader


unresolved_entity_df: DataFrame = spark.read.csv(unresolved_entity_path, header=True, inferSchema=True)

def clean_column_name(col_name: str) -> str:
    return re.sub(r'\W+', '_', col_name.strip())

unresolved_entity_df = unresolved_entity_df.toDF(*[clean_column_name(col) for col in unresolved_entity_df.columns])

unresolved_entity_df = (
  unresolved_entity_df.withColumn(embedding_source_column, F.concat_ws(" | ", *unresolved_entity_df.columns))
)

display(unresolved_entity_df)

# COMMAND ----------

from langchain_core.vectorstores.base import VectorStore
from databricks_langchain.vectorstores import DatabricksVectorSearch


columns: list[str] = spark.table(resolved_entity_table).columns

def create_vector_store() -> VectorStore:
    vector_store: VectorStore = DatabricksVectorSearch(
        index_name=vector_search_index_name,
        endpoint=vector_search_endpoint_name,
        columns=columns,
    )
    return vector_store



# COMMAND ----------

from typing import Callable
from langchain_core.language_models import LanguageModelLike
from langchain_openai import ChatOpenAI
from databricks_langchain import ChatDatabricks

api_key=os.environ["OPENAI_API_KEY"]

def create_llm() -> LanguageModelLike:
  if "OPENAI_API_KEY" in os.environ and not chat_model_name.startswith("databricks"):
    api_key=os.environ["OPENAI_API_KEY"]
    return ChatOpenAI(model=chat_model_name, api_key=api_key)
  else:
    return ChatDatabricks(endpoint=chat_model_name)



# COMMAND ----------

llm: LanguageModelLike = create_llm()
print(type(llm))

# COMMAND ----------

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from databricks_langchain import ChatDatabricks, DatabricksEmbeddings

from pydantic import BaseModel, Field


def find_candidate_matches(unresolved_document: Document, vector_store: VectorStore, k: int = 3) -> list[(Document, float)]:
    search_results: list[(Document, float)] = vector_store.similarity_search_with_score(unresolved_document.page_content, k)
    return search_results


class MatchResult(BaseModel):
  document_id: int = Field(..., description="The id of the document")
  similarity_score: float = Field(..., description="The semantic similarity returned from the vector store")
  confidence_score: float = Field(..., description="Confidence score between 0 and 1 (1 being a perfect match)")
  reason: str = Field(..., description="The reason why it was matched or not match")


def resolve_document(unresolved_document: Document, match_candidates: list[Document, float], llm: LanguageModelLike) -> MatchResult:

    candidate_prompts: list = []
    for match_candidate in match_candidates:
      candidate_prompt = f"""
      <candidate>
      Document Id: {match_candidate[0].metadata[primary_key]}
      Similarity Score: {match_candidate[1]}
      Document Metadata: {match_candidate[0].metadata}
      </candidate>
      """
      candidate_prompts.append(candidate_prompt)
    
    candidate_prompts: str = "\n".join(candidate_prompts)

    prompt = f"""
    <instructions>
    - You are an english speaking expert in entity resolution. 
    - Compare the following unresolved document to the list of candidates based on metadata and similiarity score. 
    - The simlarity score represents semantic similarity as returned from the vector store.
    - You must return the candidate and score which most closely matches the unresolved document.
    - You MUST only use the context provided. Do not use any other information.
    - Analyze and resolve the unresolved document and return the best candidate match based on the available data.
    <instructions>

    <unresolved>
    {unresolved_document.metadata}
    </unresolved>

    <candidates>
    {candidate_prompts}
    </candidates>
    """
  
    llm_with_tools = llm.with_structured_output(MatchResult)
    match_result: MatchResult = llm_with_tools.invoke(prompt)
 
    return match_result



# COMMAND ----------

from typing import Iterator

import pyspark.sql.functions as F
import pyspark.sql.types as T

import pandas as pd

from langchain_core.documents import Document


result_schema = T.StructType([
    T.StructField("source", T.StringType(), True),
    T.StructField("similarity_score", T.FloatType(), True),
    T.StructField("confidence_score", T.FloatType(), True),
    T.StructField("reason", T.StringType(), True)
])

@F.pandas_udf(result_schema)
def resolve_documents_batch(
    iterator: Iterator[pd.DataFrame]
) -> Iterator[pd.Series]:
    
    vector_store: VectorStore = create_vector_store()
    llm: LanguageModelLike = create_llm()
    #llm: LanguageModelLike = ChatOpenAI(model=chat_model_name, api_key=api_key) 


    for batch_df in iterator:
        results: list[dict[str, ...]] = []

        for _, row in batch_df.iterrows():
            
            unresolved_document: Document = Document(
                page_content=row[embedding_source_column],  
                metadata=row.to_dict()
            )
            
            match_candidates: list[Document, float] = find_candidate_matches(unresolved_document, vector_store)
            best_match_result: MatchResult = resolve_document(unresolved_document, match_candidates, llm) 
            
            matched_document: Document = None
            candidate_documents: list[Document] = [m[0] for m in match_candidates]
            for candidate_document in candidate_documents:
                candidate_id: int = int(candidate_document.metadata[primary_key])
                if candidate_id == best_match_result.document_id:
                    matched_document = candidate_document
                    break

            results.append({
                "source": matched_document.page_content,
                "similarity_score": best_match_result.similarity_score,
                "confidence_score": best_match_result.confidence_score,
                "reason": best_match_result.reason
            })

        
        yield pd.DataFrame(results)




# COMMAND ----------

import os

from pyspark.sql import DataFrame

import pyspark.sql.functions as F

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

result_df: DataFrame = ( 
  unresolved_entity_df
    .withColumn("match_result", resolve_documents_batch(F.struct(unresolved_entity_df.columns)))
)

display(result_df)

# COMMAND ----------

from langchain_community.document_loaders import DataFrameLoader

llm: LanguageModelLike = create_llm()
vector_store: VectorStore = create_vector_store()
loader = DataFrameLoader(unresolved_entity_df.toPandas(), page_content_column=embedding_source_column)
documents = loader.load()

unresolved_document = documents[0]

match_candidates: list[Document, float] = find_candidate_matches(unresolved_document, vector_store)

best_match_result: MatchResult = resolve_document(unresolved_document, match_candidates, llm)