# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC #Evaluating Risk for Loan Approvals
# MAGIC <div><img style="float:right" src="https://preview.ibb.co/jNxPym/Image.png" alt="Image" border="0"></div>
# MAGIC 
# MAGIC ## Business Value
# MAGIC 
# MAGIC 
# MAGIC Being able to accurately assess the risk of a loan application can save a lender the cost of holding too many risky assets. Rather than a credit score or credit history which tracks how reliable borrowers are, we will generate a score of how profitable a loan will be compared to other loans in the past. The combination of credit scores, credit history, and profitability score will help increase the bottom line for financial institution.
# MAGIC 
# MAGIC Having a interporable model that a loan officer can use before performing a full underwriting can provide immediate estimate and response for the borrower and a informative view for the lender

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # MLFlow - Managing the end-to-end ML lifecycle
# MAGIC 
# MAGIC <div style="float:right" ><img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/mlflow-head.png" style="height: 280px; margin:0px 0px 50px 10px"/></div>
# MAGIC 
# MAGIC 
# MAGIC * Tracking experiments to record and compare parameters and results [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html#tracking)
# MAGIC * Packaging ML code in a reusable, reproducible form in order to share with other data scientists or transfer to production. [MLflow Projects](https://mlflow.org/docs/latest/projects.html#projects)
# MAGIC * Managing and deploying models from a variety of ML libraries to a variety of model serving and inference platforms [MLflow Models](https://mlflow.org/docs/latest/models.html#models)
# MAGIC * Model registry
# MAGIC 
# MAGIC **For more information: https://mlflow.org**
# MAGIC 
# MAGIC ## From small to big ML with Databricks ML Runtime
# MAGIC 
# MAGIC Databricks ML Runtime runs optimized version of Spark ML and Horovord (deep learning) to train your models against big dataset. 
# MAGIC 
# MAGIC But small models can also benefit from Databricks and its integration with MLFlow (sklearn, keras, tensorflow...)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## The Data
# MAGIC 
# MAGIC The data used is public data from Lending Club. It includes all funded loans from 2015 to 2017. Each loan includes applicant information provided by the applicant as well as the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. For a full view of the data please view the data dictionary available [here](https://resources.lendingclub.com/LCDataDictionary.xlsx).
# MAGIC 
# MAGIC 
# MAGIC ![Loan_Data](https://preview.ibb.co/d3tQ4R/Screen_Shot_2018_02_02_at_11_21_51_PM.png)
# MAGIC 
# MAGIC https://www.kaggle.com/wendykan/lending-club-loan-data

# COMMAND ----------

from pyspark.sql.functions import *
loan_stats = spark.read.format("delta").load("/home/snehil.pandey@databricks.com/lending_data/gold/") \
                  .withColumn("loan_amnt", col("loan_amnt").cast("double"))\
                  .withColumn("annual_inc", col("annual_inc").cast("double"))\
                  .withColumn("dti", col("dti").cast("double"))\
                  .withColumn("delinq_2yrs", col("delinq_2yrs").cast("double"))\
                  .withColumn("revol_util", col("revol_util").cast("double"))\
                  .withColumn("total_acc", col("total_acc").cast("double"))\
  


loan_stats.createOrReplaceTempView("loans_gold")

# COMMAND ----------

# MAGIC %sql
# MAGIC select l.*, cast(l.loan_amnt/1000 as int)*1000 as loan_amnt_k from loans_gold l order by loan_amnt_k

# COMMAND ----------

# DBTITLE 1,Feature Distribution and Correlation
display(loan_stats)

# COMMAND ----------

# DBTITLE 1,Asset Allocation
display(loan_stats)
# display(loan_stats.groupBy("bad_loan", "grade").agg((sum(col("net"))).alias("sum_net")))

# COMMAND ----------

# DBTITLE 1,Set Response and Predictor Variables

print("------------------------------------------------------------------------------------------------")
print("Setting variables to predict bad loans")
myY = "bad_loan"
categoricals = ["term", "home_ownership", "purpose", "addr_state",
                "verification_status","application_type"]
numerics = ["loan_amnt","emp_length", "annual_inc","dti",
            "delinq_2yrs","revol_util","total_acc",
            "credit_length_in_years"]
myX = categoricals + numerics

loan_stats2 = loan_stats.select(myX + [myY, "int_rate", "net", "issue_year"])

train = loan_stats2.filter(loan_stats2.issue_year <= 2015).cache()
valid = loan_stats2.filter(loan_stats2.issue_year > 2015).cache()

# train.count()
# valid.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression Notes
# MAGIC * We will be using the Apache Spark pre-installed GLM and GBTClassifier models in this noteboook
# MAGIC * **GLM** is in reference to *generalized linear models*; the Apache Spark *logistic regression* model is a special case of a [generalized linear model](https://spark.apache.org/docs/2.2.0/ml-classification-regression.html#logistic-regression)
# MAGIC * We will also use BinaryClassificationEvaluator, CrossValidator, and ParamGridBuilder to tune our models.
# MAGIC * References to max F1 threshold (i.e. F_1 score or F-score or F-measure) is the measure of our logistic regression model's accuracy; more information can be found at [F1 score](https://en.wikipedia.org/wiki/F1_score).
# MAGIC * **GBTClassifier** is in reference to *gradient boosted tree classifier* which is a popular classification and regression method using ensembles of decision trees; more information can be found at [Gradiant Boosted Tree Classifier](https://spark.apache.org/docs/2.2.0/ml-classification-regression.html#gradient-boosted-tree-classifier)
# MAGIC * In a subsequent notebook, we will be using the XGBoost, an optimized distributed gradient boosting library.  
# MAGIC   * Underneath the covers, we will be using *XGBoost4J-Spark* - a project aiming to seamlessly integrate XGBoost and Apache Spark by fitting XGBoost to Apache Spark’s MLLIB framework.  More inforamtion can be found at [XGBoost4J-Spark Tutorial](https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_tutorial.html).

# COMMAND ----------

# DBTITLE 1,Build Grid of GLM Models w/ Standardization+CrossValidation
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.feature import StandardScaler, Imputer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

## Current possible ways to handle categoricals in string indexer is 'error', 'keep', and 'skip'
indexers = map(lambda c: StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid = 'keep'), categoricals)
ohes = map(lambda c: OneHotEncoder(inputCol=c + "_idx", outputCol=c+"_class"),categoricals)
imputers = Imputer(inputCols = numerics, outputCols = numerics)

# Establish features columns
featureCols = list(map(lambda c: c+"_class", categoricals)) + numerics

# Build the stage for the ML pipeline
# Build the stage for the ML pipeline
model_matrix_stages = list(indexers) + list(ohes) + [imputers] + \
                     [VectorAssembler(inputCols=featureCols, outputCol="features"), StringIndexer(inputCol="bad_loan", outputCol="label")]

# Apply StandardScaler to create scaledFeatures
scaler = StandardScaler(inputCol="features",
                        outputCol="scaledFeatures",
                        withStd=True,
                        withMean=True)

# Use logistic regression 
lr = LogisticRegression(maxIter=10, elasticNetParam=0.5, featuresCol = "scaledFeatures")

# Build our ML pipeline
pipeline = Pipeline(stages=model_matrix_stages+[scaler]+[lr])

# Build the parameter grid for model tuning
paramGrid = ParamGridBuilder() \
              .addGrid(lr.regParam, [0.1, 0.01]) \
              .build()

# Execute CrossValidator for model tuning
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=5)

# Train the tuned model and establish our best model
cvModel = crossval.fit(train)
glm_model = cvModel.bestModel

# Return ROC
lr_summary = glm_model.stages[len(glm_model.stages)-1].summary
display(lr_summary.roc)

#Obtain predictions on the validation set
predictions = glm_model.transform(valid)

# COMMAND ----------

# DBTITLE 1,Set Max F1 Threshold
fMeasure = lr_summary.fMeasureByThreshold
maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
maxFMeasure = maxFMeasure['max(F-Measure)']
fMeasure = fMeasure.toPandas()
bestThreshold = fMeasure[ fMeasure['F-Measure'] == maxFMeasure]
lr.setThreshold(float(bestThreshold["threshold"]))
print("Best Threshold: "+str(float(bestThreshold["threshold"]))+" for F1 score:"+str(float(bestThreshold["F-Measure"])))

# COMMAND ----------

# DBTITLE 1,Grab Model Metrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.linalg import Vectors
import mlflow 
from mlflow import spark

def extract(row):
  return (row.net,) + tuple(row.probability.toArray().tolist()) +  (row.label,) + (row.prediction,)

def score(model,data):
  pred = model.transform(data).select("net", "probability", "label", "prediction")
  pred = pred.rdd.map(extract).toDF(["net", "p0", "p1", "label", "prediction"])
  return pred 

def auc(pred):
  metric = BinaryClassificationMetrics(pred.select("p1", "label").rdd)
  return metric.areaUnderROC

glm_train = score(glm_model, train)
glm_valid = score(glm_model, valid)

glm_train.createOrReplaceTempView("glm_train")
glm_valid.createOrReplaceTempView("glm_valid")

with mlflow.start_run(run_name="glm"):
  mlflow.log_param("regParam", 0.01)
  mlflow.log_metric("auc", auc(glm_valid))
  mlflow.spark.log_model(glm_model, 'glm_model')
  
print ("GLM Training AUC:" + str(auc(glm_train)))
print ("GLM Validation AUC :" + str(auc(glm_valid)))

# COMMAND ----------

# MAGIC %md ## Quantify the Business Value
# MAGIC 
# MAGIC A great way to quickly understand the business value of this model is to create a confusion matrix.  The definition of our matrix is as follows:
# MAGIC 
# MAGIC * Prediction=1, Label=1 (Blue) : Correctly found bad loans. sum_net = loss avoided.
# MAGIC * Prediction=1, Label=0 (Orange) : Incorrectly labeled bad loans. sum_net = profit forfeited.
# MAGIC * Prediction=0, Label=1 (Green) : Incorrectly labeled good loans. sum_net = loss still incurred.
# MAGIC * Prediction=0, Label=0 (Red) : Correctly found good loans. sum_net = profit retained.
# MAGIC 
# MAGIC The following code snippet calculates the following confusion matrix.

# COMMAND ----------

# DBTITLE 1,Business Value
display(glm_valid.groupBy("label", "prediction").agg((sum(col("net"))).alias("sum_net")))

# COMMAND ----------

# MAGIC %md #What if I don't want to learn spark?
# MAGIC 
# MAGIC ### Use Koalas ! 
# MAGIC It's Pandas API backuped by Spark. Just change the pandas import by koalas and the transformation will run on spark at scale
# MAGIC 
# MAGIC ### What if my data fit in memory?
# MAGIC That's a good news! Use a single node cluster with your favorite ML framework (slearn, XGBoost)!
# MAGIC 
# MAGIC Databricks can easily distribute hyperparameter tuning on multiple node

# COMMAND ----------

#Create feature store with borrower features
#Step1: Feature computation function - The only requirement is that the comp functions must return a spark dataframe
from databricks.feature_store import feature_table

borrower_cols = ["id","addr_state","annual_inc","emp_length","home_ownership","purpose","delinq_2yrs","bad_loan"]

@feature_table
def compute_borrower_features(data):
  return data.select(borrower_cols)

borrower_df = compute_borrower_features(loan_stats)
display(borrower_df)

# COMMAND ----------

#Create feature store with loan features
from databricks.feature_store import feature_table

loan_cols = ["id","bad_loan","grade","int_rate","issue_year","loan_amnt","loan_status","purpose","term","total_pymnt","verification_status"]

@feature_table
def compute_loan_features(data):
  return data.select(loan_cols)

loan_df = compute_loan_features(loan_stats)
display(loan_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS BID_TestCase;
# MAGIC  DROP TABLE IF EXISTS BID_TestCase.borrower_df_features;
# MAGIC  DROP TABLE IF EXISTS BID_TestCase.loan_df_features;

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

borrower_df_features_table = fs.create_feature_table(
  name='BID_TestCase.borrower_df_features',
  keys='id',
  schema=borrower_df.schema,
  description='Borrower Attributes')

loan_df_features_table = fs.create_feature_table(
  name='BID_TestCase.loan_df_features',
  keys='id',
  schema=loan_df.schema,
  description='Loans Attributes')

# COMMAND ----------

compute_loan_features.compute_and_write(loan_stats, feature_table_name="BID_TestCase.loan_df_features")
compute_borrower_features.compute_and_write(loan_stats, feature_table_name="BID_TestCase.borrower_df_features")

# COMMAND ----------


