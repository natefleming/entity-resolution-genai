# Databricks notebook source
# MAGIC %pip install --upgrade --quiet langchain databricks-langchain databricks-sdk mlflow delta-spark
# MAGIC %restart_python

# COMMAND ----------

from mlflow.models import ModelConfig

config: ModelConfig = ModelConfig(development_config="model_config.yaml")

catalog_name: str = config.get("catalog_name")
database_name: str = config.get("database_name")
volume_name: str = config.get("volume_name")

resolved_entity_path: str = config.get("resolved_entity_path")
resolved_entity_table: str = config.get("resolved_entity_table")

primary_key: str = config.get("primary_key")
embedding_source_column: str = config.get("embedding_source_column")

assert catalog_name is not None
assert database_name is not None
assert volume_name is not None
assert resolved_entity_path is not None
assert resolved_entity_table is not None
assert primary_key is not None
assert embedding_source_column is not None

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

import re

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T

from delta.tables import DeltaTable, IdentityGenerator


resolved_entity_df: DataFrame = spark.read.csv(resolved_entity_path, header=True, inferSchema=True)

def clean_column_name(col_name: str) -> str:
    return re.sub(r'\W+', '_', col_name.strip())

resolved_entity_df = resolved_entity_df.toDF(*[clean_column_name(col) for col in resolved_entity_df.columns])

resolved_entity_df = resolved_entity_df.withColumn(embedding_source_column, F.concat_ws(" | ", *resolved_entity_df.columns))


(
  DeltaTable.createOrReplace(spark)
    .tableName(resolved_entity_table)
    .property("delta.enableChangeDataFeed", "true")
    .addColumn(primary_key, dataType=T.LongType(), nullable=False, generatedAlwaysAs=IdentityGenerator())
    .addColumns(resolved_entity_df.schema)
    .execute()
)

spark.sql(f"ALTER TABLE {resolved_entity_table} ADD CONSTRAINT id_pk PRIMARY KEY (id)")

resolved_entity_df.write.mode("append").saveAsTable(resolved_entity_table)

display(spark.table(resolved_entity_table))
