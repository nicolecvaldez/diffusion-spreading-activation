# -*- coding: utf-8 -*-

__author__ = "NCValdez"

import time
import os
import pandas as pd
from datetime import datetime, timedelta
from graphframes import *
from util import create_vertices, set_spark_context
from spreading_activation import SpreadingActivation

if __name__ == "__main__":
    start_time = time.time()

    # Declare date parameters
    run_date = datetime.now().strftime('%Y%m%d')

    # Launch spark context
    job_name = 'Diffusion using Spreading Activation'
    sc, sqlContext = set_spark_context(run_date, job_name)

    # Get data if in local
    edges = sqlContext.createDataFrame(pd.read_csv("data/sample.csv"))
    vertices = create_vertices(edges, "src", "dst")

    # Get data if in hdfs
    # raw_csv = sc.textFile("file://"+rel_path+"/data/sample.csv")
    # processed_csv = raw_csv.map(lambda x: (int(x[0:-1].split(",")[0]), int(x[0:-1].split(",")[1]), int(x[0:-1].split(",")[2])))
    # edges = processed_csv.toDF(["src", "dst", "weight"])
    # vertices = create_vertices(edges, "src", "dst")

    # Create graph
    g = GraphFrame(vertices, edges)

    # Start spreading activation
    sa = SpreadingActivation(g, sc, sqlContext)
    sa.attribute = "influence"
    sa.random_infected_nodes(2)
    final_g = sa.spread_activation_full(steps=5)

    # Save final graph to csv
    final_g.vertices.toPandas().to_csv("resulting_graph.csv", index=False)

    sc.stop()

    print "Ending PySpark Job - " + job_name
    print "--- %s seconds ---" % (time.time() - start_time)