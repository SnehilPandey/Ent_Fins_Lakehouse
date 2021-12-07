# Databricks notebook source
# MAGIC %md-sandbox 
# MAGIC # What we want to do: Score loan to prevent risks
# MAGIC 
# MAGIC <div style="float:right"><img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/fraud/fraud-detection-flow.png" style="height: 150px"/></div>
# MAGIC 
# MAGIC We receive demand for loan every-day, and want to be able to score based on the risk they represent.
# MAGIC 
# MAGIC Let's build a pipeline to minimize our risk!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## The Data
# MAGIC 
# MAGIC The data used is public data from Lending Club. It includes all funded loans from 2012 to 2017. Each loan includes applicant information provided by the applicant as well as the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. For a full view of the data please view the data dictionary available [here](https://resources.lendingclub.com/LCDataDictionary.xlsx).
# MAGIC 
# MAGIC 
# MAGIC ![Loan_Data](https://preview.ibb.co/d3tQ4R/Screen_Shot_2018_02_02_at_11_21_51_PM.png)
# MAGIC 
# MAGIC https://www.kaggle.com/wendykan/lending-club-loan-data

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) 1/ Cleaning Bronze data to Silver with Delta Lake
# MAGIC 
# MAGIC Delta brings Reliability (i.e. ACID transactions), performances and merge stream & batch logic

# COMMAND ----------

# MAGIC %md #### Import CSV Data and create Delta Lake Table

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/loans-1.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
loans = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(loans)

# COMMAND ----------

# DBTITLE 1,Filter Data and Fix Schema
from pyspark.sql.functions import *
loan_stats = loans.select("id", "loan_status", "int_rate", "revol_util", "issue_d", "earliest_cr_line", "emp_length", "verification_status", "total_pymnt", "loan_amnt", "grade", "annual_inc", "dti", "addr_state", "term", "home_ownership", "purpose", "application_type", "delinq_2yrs", "total_acc")

print("------------------------------------------------------------------------------------------------")
print("Create bad loan label, this will include charged off, defaulted, and late repayments on loans...")
loan_stats = loan_stats.filter(loan_stats.loan_status.isin(["Default", "Charged Off", "Fully Paid"]))\
                       .withColumn("bad_loan", (loan_stats.loan_status != "Fully Paid").cast("string"))


print("------------------------------------------------------------------------------------------------")
print("Turning string interest rate and revoling util columns into numeric columns...")
loan_stats = loan_stats.withColumn('int_rate', regexp_replace('int_rate', '%', '').cast('float')) \
                       .withColumn('revol_util', regexp_replace('revol_util', '%', '').cast('float')) \
                       .withColumn('issue_year',  substring(loan_stats.issue_d, 5, 4).cast('double') ) \
                       .withColumn('earliest_year', substring(loan_stats.earliest_cr_line, 5, 4).cast('double')) \
                       .withColumn('credit_length_in_years', (col("issue_year") - col("earliest_year")))


print("------------------------------------------------------------------------------------------------")
print("Converting emp_length column into numeric...")
loan_stats = loan_stats.withColumn('emp_length', trim(regexp_replace(col("emp_length"), "([ ]*+[a-zA-Z].*)|(n/a)", "") )) \
                       .withColumn('emp_length', trim(regexp_replace(col("emp_length"), "< 1", "0") )) \
                       .withColumn('emp_length', trim(regexp_replace(col("emp_length"), "10\\+", "10") ).cast('float'))


# COMMAND ----------

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Paths for various Delta tables
bronze_tbl_path = '/home/{}/lending/bronze/'.format(user)
silver_tbl_path = '/home/{}/lending/silver/'.format(user)
merge_tbl_path =  '/home/{}/lending/merge/'.format(user)
gold_tbl_path  = '/home/{}/lending/gold/'.format(user)
automl_tbl_path = '/home/{}/lending/automl-silver/'.format(user)
lending_preds_path = '/home/{}/lending/preds/'.format(user)


bronze_tbl_name = 'bronze_lending'
silver_tbl_name = 'silver_lending'
merge_tbl_name = 'merge_lending'
gold_tbl_name = 'gold_lending'
automl_tbl_name = 'automl_lending'
lending_preds_tbl_name = 'lending_preds'

# COMMAND ----------

# Set config for database name, file paths, and table names
database_name = 'lending'

# Delete the old database and tables if needed
_ = spark.sql('DROP DATABASE IF EXISTS {} CASCADE'.format(database_name))

# Create database to house tables
_ = spark.sql('CREATE DATABASE {}'.format(database_name))

# COMMAND ----------

# Save table as Delta Lake
loan_stats.write.format("delta").mode("overwrite").save(silver_tbl_path)

# COMMAND ----------

display(loan_stats)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS lending.silver_lending

# COMMAND ----------

# Create silver table
silver_table = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,silver_tbl_name,silver_tbl_path))

# COMMAND ----------

# DBTITLE 1,Run SQL Queries on top of our data
# MAGIC %sql 
# MAGIC -- View Delta Lake table
# MAGIC SELECT * FROM lending.silver_lending

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Delta has a full DML Support (UPDATE/DELETE/MERGE):
# MAGIC 
# MAGIC Let's delete all our data from the state of Texas per requirement from our GDPR officer

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Review current loans within the `loan_by_state_delta` Delta Lake table
# MAGIC select addr_state, count(*) as loans from lending.silver_lending group by addr_state

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM lending.silver_lending WHERE addr_state='TX';
# MAGIC 
# MAGIC select addr_state, count(*) as loans from lending.silver_lending group by addr_state

# COMMAND ----------

# MAGIC %md ##Going back in time... 

# COMMAND ----------

# MAGIC %sql DESCRIBE HISTORY lending.silver_lending

# COMMAND ----------

# DBTITLE 1,SQL Merge: Upsert your data
# Let's create a simple table to merge
merge_csv =spark.read.format("csv").option("header",True).load("/FileStore/tables/merge.csv")

# Save table as Delta Lake
merge_csv.write.format("delta").mode("overwrite").save(merge_tbl_path)
merge_table = spark.read.format("delta").load(merge_tbl_path)
merge_table.createOrReplaceTempView("merge_table")
display(merge_table)

# COMMAND ----------

# MAGIC %md Instead of writing separate `INSERT` and `UPDATE` statements, we can use a `MERGE` statement. 

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO lending.silver_lending as d
# MAGIC USING merge_table as m
# MAGIC on d.id = m.id
# MAGIC WHEN MATCHED THEN 
# MAGIC   UPDATE SET *
# MAGIC WHEN NOT MATCHED 
# MAGIC   THEN INSERT *

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC ##![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Let's create our GOLD final data, adding some extra information in our final table:
# MAGIC 
# MAGIC <div style="float:right"><img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/fraud/fraud-detection-flow.png" style="height: 150px"/></div>

# COMMAND ----------

print("------------------------------------------------------------------------------------------------")
print("Map multiple levels into one factor level for verification_status...")
loan_stats_gold = spark.read.format("delta").load(silver_tbl_path).withColumn('verification_status', trim(regexp_replace(col('verification_status'), 'Source Verified', 'Verified')))

print("------------------------------------------------------------------------------------------------")
print("Calculate the total amount of money earned or lost per loan...")
loan_stats_gold = loan_stats_gold.withColumn('net', round( col('total_pymnt') - col('loan_amnt'), 2))
loan_stats_gold.write.mode("overwrite").format("delta").save(gold_tbl_path)

# COMMAND ----------

# DBTITLE 1,Create Gold Table
# Create gold table
gold_table = spark.sql('''
  CREATE TABLE `{}`.{}
  USING DELTA 
  LOCATION '{}'
  '''.format(database_name,gold_tbl_name,gold_tbl_path))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM lending.gold_lending

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) What else can we do with Delta ?
# MAGIC 
# MAGIC ####Reliability
# MAGIC - Schema Enforcement
# MAGIC - Schema Evolution
# MAGIC - Merge Batch & Streaming
# MAGIC - Travel Back in time
# MAGIC 
# MAGIC ####Performances
# MAGIC - Compact small files (automatically)
# MAGIC - Z-ORDER (index to increase query time)
# MAGIC - Delta Cache (cache data locally, enabled by default!)

# COMMAND ----------

#Merge Batch & Streaming
#Delta Cache (cache data locally, enabled by default!)
#add photon and non photon perf test 
#add autoloader with unification, time travel 
#language supported
#collaborative
#IDE integration
#simplicity 

# COMMAND ----------


