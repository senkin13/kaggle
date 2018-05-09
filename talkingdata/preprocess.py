## spark.sql
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test").enableHiveSupport().getOrCreate()
query = spark.sql('''
SELECT
ip,
app,
device,
os,
channel,
day,
hour,
minute,
is_attributed,
click_time
FROM 
default.td
ORDER BY click_time
''')

import time
query.coalesce(1).write.option("header","true").csv('/tmp/kaggle/features/' + str(time.time()), compression='gzip')

spark.stop()



##spark submit
spark-submit --master yarn --driver-memory 8g --num-executors 10 --executor-memory 5g --executor-cores 1 --queue default --conf spark.yarn.executor.memoryOverhead=1000 $1
