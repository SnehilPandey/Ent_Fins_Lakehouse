-- Databricks notebook source
SHOW DATABASES 

-- COMMAND ----------

USE lending

-- COMMAND ----------

SHOW TABLES

-- COMMAND ----------

-- DBTITLE 1,Total Loan Amount
SELECT
  SUM(loan_amnt) as total_loan_amount
FROM
  lending.gold_lending

-- COMMAND ----------

-- DBTITLE 1,Purpose of loan
select
  purpose,
  sum(loan_amnt) as total_loan
from
  lending.gold_lending
group by
  purpose

-- COMMAND ----------

-- DBTITLE 1,Total Loan Amount
select
  addr_state,
  count(id) as number_of_loans
from
  lending.gold_lending
group by
  addr_state,
  verification_status
