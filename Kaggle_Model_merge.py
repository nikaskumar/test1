//val items = spark.read.option("header","True").option("inferSchema","True").csv("/tm/kaggle/data/items.csv")
val stores = spark.read.option("header","True").option("inferSchema","True").csv("/tm/kaggle/data/stores.csv")
val txn = spark.read.option("header","True").option("inferSchema","True").csv("/tm/kaggle/data/transactions.csv")
//val oil = spark.read.option("header","True").option("inferSchema","True").csv("/tm/kaggle/data/oil.csv")
val hl_en = spark.read.option("header","True").option("inferSchema","True").csv("/tm/kaggle/data/holidays_events.csv.csv")
val train = spark.read.option("header","True").option("inferSchema","True").csv("/tm/kaggle/data/train.csv")


//Make data 
import org.apache.spark.sql.functions.{concat, lit}
import org.apache.spark.ml.feature.VectorAssembler
//import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
//Take Data Only after first April 
val train_1  = train.filter("date >= '2017-04-01 00:00:00'").cache()
//Create a new_ID column as combination of Store and ID
val train_2 = train_1.withColumn("str_itm", concat($"store_nbr",lit("_"),$"item_nbr")).select("str_itm","item_nbr","date","unit_sales")
//Values on which data is to be pivot
val sqn = train_2.select("date").distinct.rdd.map(row =>row(0).asInstanceOf[java.sql.Timestamp]).collect.toSeq.sortWith(_.getTime < _.getTime)
//Pivot the data
val train_3 = train_2.groupBy("str_itm","item_nbr").pivot("date",sqn).max("unit_sales").na.fill(0).withColumnRenamed("2017-08-15 00:00:00","target")

//Renamed Columns
val cols = train_3.columns.zipWithIndex.map(x=>{if(!(Array(1,2,(sqn.length+2))).contains((x._2+1))) {"var"+(x._2-1)} else{x._1} }).toSeq 
//Rename Columns as Currentnly they are as Date
val train_4 = train_3.toDF(cols: _*).cache()

val train_5 = train_4.withColumn("id", monotonically_increasing_id())



val items = spark.read.option("header","True").option("inferSchema","True").csv("/tm/kaggle/data/items.csv")

val sq_it = items.select("family").distinct.rdd.map(row =>row(0).asInstanceOf[String]).collect.toSeq
val item_1 = items.groupBy("item_nbr","perishable").pivot("family",sq_it).count().na.fill(0)
val it_cols = item_1.columns.zipWithIndex.map(x=>{if(!(Array(0,1)).contains((x._2))) {"prod"+(x._2-1)} else{x._1} }).toSeq 
//Rename Columns as Currentnly they are as Date
val item_2 = item_1.toDF(it_cols: _*).cache()
val tem_train = train_5.selectExpr("id","str_itm","item_nbr as itm1")

val itm_3 = tem_train.join(item_2, tem_train.col("itm1")=== item_2.col("item_nbr"), "left").drop("itm1")


//val trn = train_4.withColumn("id", monotonically_increasing_id())
val colnames = train_5.columns diff Array("str_itm","target","id","item_nbr")
val assembler = new VectorAssembler().setInputCols(colnames).setOutputCol("features")
val df_1 = assembler.transform(train_5).select("id","str_itm","item_nbr","target","features").orderBy("id")
df_1.write.mode("overwrite").parquet("/tm/kaggle/data/train_sale06.parquet")

//val item_4 = itm_3.withColumn("id1", monotonically_increasing_id())
val colnames = itm_3.columns diff Array("str_itm","item_nbr","id")
val assembler = new VectorAssembler().setInputCols(colnames).setOutputCol("features")
val df_1 = assembler.transform(itm_3).select("id","str_itm","item_nbr","features").orderBy("id")
df_1.write.mode("overwrite").parquet("/tm/kaggle/data/train_prod06.parquet")

=============================================================================================================
from pyspark.ml.feature import VectorAssembler
import numpy as np
from pyspark.sql.functions import col
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import math
from keras.utils import np_utils
from numpy import argmax
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorIndexer
import readline, rlcompleter
readline.parse_and_bind("tab:complete")
#=============================================================================

trn = spark.read.parquet('/tm/kaggle/data/train_sale06.parquet').orderBy("id")
colnames = ["id","target","features"]
assembler=VectorAssembler(inputCols=colnames,outputCol='features1')
output = assembler.transform(trn).select('features1')


dataframe = output.rdd.map(lambda x : x.features1.toArray())
dataframe1 = dataframe.collect()
arr = np.array(dataframe1, dtype=np.float64)
ln=arr.shape[0]

z=[]
y=[]
x=[]
for i in range(ln):
              z.append(arr[i][0])
              y.append(arr[i][1])
              x.append(arr[i][2:])

train_id=np.array(z)
trainX=np.array(x)
trainY=np.array(y)


# TEST DATA

output = assembler.transform(trn).select('features1')
dataframe = output.rdd.map(lambda x : x.features1.toArray())
dataframe1 = dataframe.collect()
test_arr = np.array(dataframe1, dtype=np.float64)
ln=test_arr.shape[0]

z=[]
y=[]
x=[]
for i in range(ln):
              z.append(test_arr[i][0])
              y.append(test_arr[i][1])
              x.append(test_arr[i][2:])

test_id=np.array(z)
testX=np.array(x)
testY=np.array(y)
#==============Item/Product Data For Model================================
itm = spark.read.parquet('/tm/kaggle/data/train_prod06.parquet').orderBy("id") 
it_cols = ["id","features"]
it_assem=VectorAssembler(inputCols=it_cols,outputCol='features1')
it_out = it_assem.transform(itm).select('features1')
it_datf = it_out.rdd.map(lambda x : x.features1.toArray())
it_datf1 = it_datf.collect()
arr = np.array(it_datf1, dtype=np.float64)
ln=arr.shape[0]

z=[]
x=[]
for i in range(ln):
              z.append(arr[i][0])
              x.append(arr[i][1:])
              

itm_id=np.array(z)
itmX=np.array(x)
 


# KERAS
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.layers import Merge


'''model = Sequential()
model.add(Dense(250, input_dim=trainX.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=200)'''


# Model1 is for Produt Features
model1 = Sequential()
model1.add(Dense(30, input_dim=itmX.shape[1], activation='relu'))
model1.add(Dense(10,activation='relu'))
model1.add(Dense(1,activation='relu'))

#Model2 is for Temporal Data 
model2 = Sequential()
model2.add(Dense(250, input_dim=trainX.shape[1], kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, activation='relu'))
model2.add(Dense(100, activation='relu'))
model2.add(Dense(1, activation='relu'))

#Merge two models to get final model
merged = Merge([model1, model2], mode='concat')
final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(1, activation='relu'))
final_model.compile(optimizer='adam', loss='mean_squared_error')
final_model.fit([itmX, trainX], trainY,  epochs=40, batch_size=200)

# make predictions
testPredict = final_model.predict([itmX, testX])


#Make Prediction For next 17 Days of Test data 
for i in range(17):
  pred = final_model.predict([itmX, testX])
  testX = np.concatenate((testX,pred), axis = 1)
  testX = np.delete(testX, 0, axis=1)



#Transform Predicted Numpy Array data in data Frame format
pred_17 = np.roll(testX,17,axis = 1)[:,:17]
predicted = np.concatenate((test_id.reshape((testX.shape[0], 1)),pred_17),axis =1)
coln_t = ['id_t'] + ['aug'+str(i) for i in range(15,32)]
df_pred = pd.DataFrame(predicted, columns=coln_t)

dd = sqlContext.createDataFrame(df_pred).cache()

from pyspark.sql.types import DoubleType
test_s = trn.withColumn("nn_id", trn["id"].cast(DoubleType())).drop("id").withColumnRenamed("nn_id","id")
new_test = test_s.join(dd, test_s.id==dd.id_t).drop("id_t").drop("target").drop("features").drop("id").drop("aug15").drop("item_nbr")


#To Melt the data We leveraged Pandas
nn_tst1 = new_test.toPandas()
nn_tst2 = pd.melt(nn_tst1, id_vars=['str_itm'], value_vars=list(nn_tst1.columns[1:]))
nn_tst2['value'] = np.where(nn_tst2['value'] >=0, nn_tst2['value'], 0.0)
sqlContext.createDataFrame(nn_tst2).write.mode("overwrite").parquet("/tm/kaggle/data/nn_mdl_temp3.parquet")




======================Merge Data Scala===================================================================
import org.apache.spark.sql.functions.{concat, lit}
import org.apache.spark.sql.types._
val test = spark.read.option("header","True").option("inferSchema","True").csv("/tm/kaggle/data/test.csv")
val sam_sub = spark.read.option("header","True").option("inferSchema","True").csv("/tm/kaggle/data/sample_submission.csv")
val md_file = spark.read.parquet("/tm/kaggle/data/nn_mdl_temp3.parquet")

val test_1 = test.withColumn("str_itt", concat($"store_nbr",lit("_"),$"item_nbr")).select("id","str_itt","date").withColumn("nn", concat($"str_itt", lit("_"), substring($"date".cast(StringType),9,2))).cache()
val md_fl1 = md_file.withColumn("nn_id", concat($"str_itm",lit("_"),substring($"variable",4,2))) 
val join_1 = test_1.select("id","nn").join(md_fl1, test_1.col("nn")===md_fl1.col("nn_id"), "left").selectExpr("id as id1","value").na.fill(0).cache
val join_2 = sam_sub.join(join_1, sam_sub.col("id")===join_1.col("id1"), "left").drop("unit_sales").drop("id1").withColumnRenamed("value", "unit_sales")
join_2.coalesce(1).write.mode("overwrite").option("header", "true").csv("/tm/kaggle/data/nn_subm3.csv")
