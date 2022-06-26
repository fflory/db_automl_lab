# Databricks notebook source
# MAGIC %md # AutoML Lab
# MAGIC 
# MAGIC [Databricks AutoML](https://docs.databricks.com/applications/machine-learning/automl.html) helps you automatically build machine learning models both through a UI and programmatically. It prepares the dataset for model training and then performs and records a set of trials (using HyperOpt), creating, tuning, and evaluating multiple models. 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Use AutoML to automatically train and tune your models
# MAGIC  - Run AutoML in Python and through the UI
# MAGIC  - Interpret the results of an AutoML run
# MAGIC  - Deploy a model in production using the Model Registry
# MAGIC  - Run a batch inference using the Production model

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md Currently, AutoML uses a combination of XGBoost and sklearn (only single node models) but optimizes the hyperparameters within each.

# COMMAND ----------

file_path = f"{datasets_dir}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md ### Use the UI
# MAGIC 
# MAGIC Instead of programmatically building our models, we can also use the UI. But first we need to register our dataset as a table.

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {cleaned_username}")
train_df.write.mode("overwrite").saveAsTable(f"{cleaned_username}.autoMLTable")

print(f"Your dataset is saved under the table: {cleaned_username}.autoMLTable")

# COMMAND ----------

# MAGIC %md First, make sure that you have selected the Machine Learning role on the left, before selecting start AutoML on the workspace homepage.
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/301/AutoML_1_2.png" alt="step12" width="750"/>

# COMMAND ----------

# MAGIC %md Select `regression` as the problem type, as well as the table we created in the previous cell. Then, select `price` as the column to predict.
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/301/AutoML_UI.png" alt="ui" width="750"/>

# COMMAND ----------

# MAGIC %md In the advanced configuration dropdown, change the evaluation metric to rmse, timeout to 5 minutes, and the maximum number of runs to 20.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/301/AutoML_Advanced.png" alt="advanced" width="500"/>

# COMMAND ----------

# MAGIC %md Finally, we can start our run. 
# MAGIC 
# MAGIC Once completed, the first thing to do is to analyze the data viewing the data exploration notebook
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-1.1.png" alt="results" width="1000"/>

# COMMAND ----------

# MAGIC %md 
# MAGIC We can then review the best run notebook and understand which model has been used.
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-1.2.png" alt="results" width="1000"/>

# COMMAND ----------

# MAGIC %md ## Saving the best model in MLFlow repository

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's deploy our best model in production. To do that, select the best run from your autoML experiment
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-2.png" alt="results" width="1000"/>

# COMMAND ----------

# MAGIC %md 
# MAGIC This will bring you to the experiment detail. You can see the parameters and metrics of this specific run. Note that the notebook used to do this run is linked, this gives you reproducibility and traceability over your ML experiment.
# MAGIC 
# MAGIC The next step is to send this model in the MLFlow Registry:
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-3.png" alt="results" width="1000"/>

# COMMAND ----------

# MAGIC %md
# MAGIC Save this model under `automl_lab`. If the model already exists with this name, you can simply save it as a new version. 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-4.png" alt="results" width="500"/>
# MAGIC 
# MAGIC Congrats! your model is now in the MLFlow registry!

# COMMAND ----------

# MAGIC %md ## Deploying this model in production

# COMMAND ----------

# MAGIC %md
# MAGIC Let's deploy your model in production. Click on Models, search the `automl_lab` model, and select the last version you just uploaded.
# MAGIC 
# MAGIC You can now change the stage, and transition it to Production directly. Add a comment and save.
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-5.png" alt="results" width="1000"/>
# MAGIC 
# MAGIC Congrats! Your model is now flagged as production ready!

# COMMAND ----------

# MAGIC %md ## Running inferences using your model

# COMMAND ----------

# MAGIC %md
# MAGIC Let's now load our model from MLFlow registry to run inferences in our entire table.
# MAGIC 
# MAGIC In your model, click on "Use model for inference". 
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-6.png" alt="results" width="1000"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Databricks offers Batch inference and real-time (over REST API). We'll deploy this model using a batch mode (ex: score all our dataset every night)
# MAGIC 
# MAGIC Select the "Production" model, it'll automatically select the most up to date model. As input table, select the dataset you used for the auto-ml training.
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/auto-ml/auto-ml-labs-7.png" alt="results" width="500"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Congrats!
# MAGIC 
# MAGIC You now have a notebook that you can use to run inferences at scale on your entire dataset. We could now easily schedule this notebook to run every night, or even deploy the model as part of a DLT pipeline.
# MAGIC 
# MAGIC Note that the generated inference notebook is library-agnostic. The model we load could be from sklearn or xgboost, MLFlow abstract that away to simplify our deployment.
# MAGIC 
# MAGIC Take some time to review and run inference notebook, and upload a screenshot of the last part of the inference notebook as proof of completion of your Auto-ML labs!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
