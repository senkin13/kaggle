## Import train/test_supplement to Hive table
CREATE TABLE IF NOT EXISTS train(
ip int,
app int,
device int,
os int,
channel int,
click_time string,
attributed_time string,
is_attributed int)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ‘,’
LINES TERMINATED BY ‘\n’
STORED AS TEXTFILE
tblproperties (“skip.header.line.count”=“1");


CREATE TABLE IF NOT EXISTS test(
click_id int,
ip int,
app int,
device int,
os int,
channel int,
click_time string)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ‘,’
LINES TERMINATED BY ‘\n’
STORED AS TEXTFILE
tblproperties (“skip.header.line.count”=“1");

               
LOAD DATA INPATH '/tmp/test_supplement.csv' INTO TABLE test;
LOAD DATA INPATH '/tmp/train.csv' INTO TABLE train;

               
CREATE TABLE IF NOT EXISTS td(
ip int,
app int,
device int,
os int,
channel int,
day int,
hour int,
minute int,
click_time string,
is_attributed string)               
PARTITIONED BY(dt string)
ROW FORMAT DELIMITED               
FIELDS TERMINATED BY ','               
LINES TERMINATED BY '\n'               
STORED AS PARQUET               
TBLPROPERTIES ("orc.compress"="SNAPPY");
               
               
SET hive.exec.dynamic.partition.mode=nonstrict;
SET hive.exec.dynamic.partition=true;
SET hive.optimize.sort.dynamic.partition=true;
INSERT INTO TABLE td PARTITION(dt) SELECT ip,app,device,os,channel,
               day(from_unixtime(unix_timestamp(click_time)+28800)) as day,
               hour(from_unixtime(unix_timestamp(click_time)+28800)) as hour,
               minute(from_unixtime(unix_timestamp(click_time)+28800)) as minute,
               from_unixtime(unix_timestamp(click_time)+28800) as click_time,
               from_unixtime(unix_timestamp(attributed_time)+28800) as attributed_time,
               is_attributed,
               TO_DATE(from_unixtime(unix_timestamp(click_time)+28800))
               FROM train
               UNION
               SELECT ip,app,device,os,channel,
               day(from_unixtime(unix_timestamp(click_time)+28800)) as day,
               hour(from_unixtime(unix_timestamp(click_time)+28800)) as hour,
               minute(from_unixtime(unix_timestamp(click_time)+28800)) as minute,
               from_unixtime(unix_timestamp(click_time)+28800) as click_time,
               ,
               ,
               TO_DATE(from_unixtime(unix_timestamp(click_time)+28800))
               FROM test;

               
## Spark.sql
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
td
ORDER BY click_time
''')

import time
query.coalesce(1).write.option("header","true").csv('/tmp/kaggle/features/' + str(time.time()), compression='gzip')

spark.stop()



##Spark submit
spark-submit --master yarn --driver-memory 8g --num-executors 10 --executor-memory 5g --executor-cores 1 --queue default --conf spark.yarn.executor.memoryOverhead=1000 $1
