# -*- coding: utf-8 -*-

from dateutil.relativedelta import relativedelta
import py4j
import pyspark
from pyspark import SparkConf, SparkContext, SparkFiles
from pyspark.sql import SQLContext, HiveContext
from pyspark import SparkFiles
from StringIO import StringIO
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import *
from pyspark.sql.types import *

def create_vertices(edges, src_col, dst_col):
    """
    From graph edges data, create list of vertices.

    :param data: pyspark dataframe, edge data containing source, destination and weight columns
    :param src_col: str, source column name in data
    :param dst_col: str, destination column name in data

    :return: pyspark dataframe, containing vertices data with column name : "id"
    """

    data_1 = edges.select(src_col)
    data_2 = edges.select(dst_col)
    vertices = data_1.unionAll(data_2).dropDuplicates()
    exp = "%s as id" %(src_col)

    return vertices.selectExpr(exp)


def set_spark_context(rundate, appname):
    conf = SparkConf().\
            setAppName(appname + " " + str(rundate)).\
            set('spark.hadoop.mapreduce.output.fileoutputformat.compress', 'false').\
            set('spark.sql.parquet.compression.codec','uncompressed')
    sc = SparkContext(conf=conf)
    try:
        sc._jvm.org.apache.hadoop.hive.conf.HiveConf()
        sqlCtx = sqlContext = HiveContext(sc)
    except py4j.protocol.Py4JError:
        sqlCtx = sqlContext = SQLContext(sc)
    return sc, sqlContext