# Databricks notebook source
# MAGIC %md ## Databricks Auto Loader demo
# MAGIC 
# MAGIC Auto Loader incrementally and efficiently processes new data files as they arrive in storage.
# MAGIC 
# MAGIC Auto Loader provides a Structured Streaming source called `cloudFiles`. Given an input directory path on the cloud file storage, the `cloudFiles` source automatically processes new files as they arrive, with the option of also processing existing files in that directory.
# MAGIC 
# MAGIC A pipeline using the Auto Loader can be deployed in 2 primary modes:
# MAGIC * Always-on streaming job: Constantly running Databricks Job which picks up new data immediately and processes it.
# MAGIC * Scheduled one-shot job: Periodic Databricks Jobs which looks for new data and processes it.
# MAGIC 
# MAGIC Resources
# MAGIC * Documentation: [AWS](https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html#load-files-from-s3-using-auto-loader), [Azure](https://docs.microsoft.com/en-us/azure/databricks/spark/latest/structured-streaming/auto-loader)
# MAGIC * Blog post introducing the Auto Loader: [Introducing Databricks Ingest: Easy and Efficient Data Ingestion from Different Sources into Delta Lake](https://databricks.com/blog/2020/02/24/introducing-databricks-ingest-easy-data-ingestion-into-delta-lake.html)
# MAGIC * Blog post with a detailed example use case: [How to build a Quality of Service (QoS) analytics solution for streaming video services](https://databricks.com/blog/2020/05/06/how-to-build-a-quality-of-service-qos-analytics-solution-for-streaming-video-services.html)

# COMMAND ----------

# MAGIC %md ### Demo overview
# MAGIC 
# MAGIC * Create initial JSON files in a `raw` directory.
# MAGIC * Run an Auto Loader job to ingest the initial files into the Delta table.
# MAGIC * Create new JSON files in the `raw` directory.
# MAGIC * Run the same Auto Loader job to ingest those new files, merging them into the existing Delta table.
# MAGIC * Running the Auto Loader job once more --- but with no new files --- does nothing.

# COMMAND ----------

# MAGIC %md Create initial JSON files in a `raw` directory.

# COMMAND ----------

initialData = spark.createDataFrame(
  [('a', 1),
   ('b', 2),
   ('c', 3),
   ('d', 4),
  ]).toDF('label', 'count')
display(initialData)

# COMMAND ----------

temp_dir = 'dbfs:/tmp/joseph/autoloader'
local_temp_dir = temp_dir.replace('dbfs:', '/dbfs')
dbutils.fs.mkdirs(temp_dir + '/raw')
dbutils.fs.mkdirs(temp_dir + '/checkpoint')

# COMMAND ----------

initialPandas = initialData.toPandas()
initialPandas.to_json(local_temp_dir + '/raw/data1.json', orient='records')

# COMMAND ----------

# MAGIC %sh cat /dbfs/tmp/joseph/autoloader/raw/data1.json

# COMMAND ----------

# MAGIC %md Run an Auto Loader job to ingest the initial files into the Delta table.

# COMMAND ----------

schema = initialData.schema
schema

# COMMAND ----------

def auto_loader_job(schema):
  input_df = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.includeExistingFiles", "true")
    .schema(schema)
    .load(temp_dir + "/raw")
  )
  (
    input_df.writeStream
    .trigger(once=True)
    .format("delta")
    .option("checkpointLocation", temp_dir + "/checkpoint")
    .option("path", temp_dir + "/bronze.delta")
    .start()
  )

# COMMAND ----------

auto_loader_job(schema)

# COMMAND ----------

display(spark.read.format("delta").load(temp_dir + "/bronze.delta"))

# COMMAND ----------

# MAGIC %md Create new JSON files in the `raw` directory.

# COMMAND ----------

nextData = spark.createDataFrame(
  [('e', 5),
   ('f', 6),
   ('g', 7),
   ('h', 8),
  ]).toDF('label', 'count')
nextPandas = nextData.toPandas()
nextPandas.to_json(local_temp_dir + '/raw/data2.json', orient='records')

# COMMAND ----------

# MAGIC %md Run the same Auto Loader job to ingest those new files, merging them into the existing Delta table.

# COMMAND ----------

auto_loader_job(schema)

# COMMAND ----------

display(spark.read.format("delta").load(temp_dir + "/bronze.delta"))

# COMMAND ----------

# MAGIC %md Running the Auto Loader job once more --- but with no new files --- does nothing.

# COMMAND ----------

auto_loader_job(schema)

# COMMAND ----------

display(spark.read.format("delta").load(temp_dir + "/bronze.delta"))

# COMMAND ----------

# Clean up temp directory
dbutils.fs.rm(temp_dir, recurse=True)

# COMMAND ----------


