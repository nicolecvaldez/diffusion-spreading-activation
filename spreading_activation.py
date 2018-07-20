# -*- coding: utf-8 -*-

from pyspark.sql.functions import sum as sqlsum
from pyspark.sql.functions import *
from graphframes.lib import AggregateMessages as AM
from graphframes import GraphFrame
from pyspark.sql import SQLContext, functions as sqlfunctions, types


class SpreadingActivation(object):
    """
    - A model that represents a “word-of-mouth” scenario where a node influences one of his neighbors, from where
    the influence spreads to some other neighbor, and so on.
    - At the end of the diffusion process, we inspect the amount of influence received by each node. Using a
    threshold-based technique, a node that is currently not influenced can be declared to be a potential future one,
    based on the influence that has been accumulated.
    - The diffusion model is based on Spreading Activation (SPA) techniques proposed in cognitive psychology
    and later used for trust metric computations.
    - For more details, please see paper entitled "Social Ties and their Relevance to Churn in Mobile Telecom Networks"
    (https://pdfs.semanticscholar.org/3275/3d80adb5ec2d4a974b5d1a872e2c957b263b.pdf)
    """

    def __init__(self, graph, sc=None, sqlContext=None):

        self.sc = sc
        self.sqlContext = sqlContext

        # - Allowed values: pyspark graphframe object
        # - Contains graph/network with nodes as objects and edges as relationships between objects
        self.graph = graph

        # - Allowed values: python list or pyspark dataframe
        # - Collection of nodes that is "infected" or is the source of influence
        self.infected_nodes = None

        # - Allowed values: string (column name)
        # - Column name which will store amount of influence transfer
        self.attribute = "influence"

        # - Allowed values: float (0 - 1)
        # - Percentage of influence to distribute
        # - The spreading factor determines the amount of importance we wish to associate on the
        # distance of an active node from the initial seed node(s). Low values of d favor
        # influence proximity to the source of injection, while high values allow the influence
        # to also reach nodes which are further away.
        self.spreading_factor = 0.60

        # - Allowed values: string ("weighted" or "unweighted")
        # - Once a node decides what fraction of energy to distribute, the next step is to decide what
        # fraction of the energy is transferred to each neighbor. This is controlled by a Transfer Function F.
        # - weighted: Energy distributed along the directed edge <X,Y> depends on its relative weight compared
        # to the sum of weights of all outgoing edges of X.
        # - unweighted: Energy distributed along the edge <X,Y> independent of its relative weight
        self.transfer_function = "weighted"

        # - Allowed values: float (0 - 1)
        # - One of two termination conditions: change in influence is not greater than accuracy threshold
        # - Only activated when steps = 0 in spread_activation_full function
        # - WARNING: Termination of iteration not guaranteed
        # - Since the directed call graph contains cycles, the computation of influence values for all
        # reachable nodes is inherently recursive if the number of iteration (steps) is not specified.
        # The accuracy threshold is one of the termination condition where changes in influence is
        # not greater than accuracy threshold.
        self.accuracy_threshold = 0.01

    def set_infected_nodes(self, list_or_dataframe):
        """
        Set nodes that is infected or is the source of influence using pyspark dataframe.
        :param dataframe: pyspark dataframe with column 'id' or python list
        :return:
        """

        infected_dataframe = list_or_dataframe

        # Convert list to dataframe
        if type(list_or_dataframe) == list:
            rdd_list = self.sc.parallelize(list_or_dataframe)
            row_rdd_list = rdd_list.map(lambda x: Row(x))
            field_list = [StructField("id", LongType(), True)]
            schema_list = StructType(field_list)
            infected_dataframe = self.sqlContext.createDataFrame(row_rdd_list, schema_list)

        # Create column for influence attribute containing 1's
        infected_dataframe = infected_dataframe.withColumn(self.attribute, lit(1.0))
        infected = infected_dataframe

        self.infected_nodes = infected_dataframe

        # Merge to original vertices of graph
        orig_vertices = self.graph.vertices.selectExpr("id as id")

        # Update graph
        orig_edges = self.graph.edges
        new_vertices = orig_vertices.join(infected, "id", "left_outer").na.fill(0)
        self.graph = GraphFrame(new_vertices, orig_edges)

    def random_infected_nodes(self, n_nodes):
        """
        Randomly set nodes that is infected or is the source of influence using the number of infected nodes input.
        :param n_nodes: int, number of nodes to infect randomly
        :return:
        """

        # Randomly get n_nodes number of nodes
        random_sample = self.graph.vertices.rdd.takeSample(False, n_nodes)
        randomly_infected_nodes = self.sqlContext.createDataFrame(random_sample)

        # Set infected nodes
        self.set_infected_nodes(randomly_infected_nodes)

    def compute_degrees(self, graph):
        """
        Compute weighted and unweighted in and out degrees in graph. Re-declares graph to add the following
        attributes: inDegree, outDegree, w_inDegree, w_outDegree.
        :param graph: graphframe object, network
        :return:
        """

        g_vertices = graph.vertices
        g_edges = graph.edges

        # Get unweighted degrees
        indeg = graph.inDegrees
        outdeg = graph.outDegrees

        # Get weighted degrees
        w_indeg = (g_edges.groupby("dst").agg(sum("weight").alias("w_inDegree"))).selectExpr("dst as id",
                                                                                             "w_inDegree as w_inDegree")
        w_outdeg = (g_edges.groupby("src").agg(sum("weight").alias("w_outDegree"))).selectExpr("src as id",
                                                                                               "w_outDegree as w_outDegree")
        # Update vertices attribute
        new_v = g_vertices.join(indeg, "id", "left_outer")
        new_v = new_v.join(outdeg, "id", "left_outer")
        new_v = new_v.join(w_indeg, "id", "left_outer")
        new_v = new_v.join(w_outdeg, "id", "left_outer")
        new_v = new_v.na.fill(0)

        # Update graph
        self.graph = GraphFrame(new_v, g_edges)

    def spread_activation_step(self, graph, attribute, spreading_factor, transfer_function):
        """
        One step in the spread activation model.
        :param graph: graphframe object, network
        :param attribute: str, name of attribute/influence
        :param spreading_factor: 0 - 1, amount of influence to spread
        :param transfer_function: weighted or unweighted, how to transfer influence along edges
        :return: graphframe object, new network with updated new calculation of attribute in vertices
        """

        # Pass influence/message to neighboring nodes (weighted/unweighted option)
        if transfer_function == "unweighted":
            msgToSrc = (AM.src[attribute] / AM.src["outDegree"]) * (1 - spreading_factor)
            msgToDst = sqlfunctions.when(AM.dst["outDegree"] != 0,
                                         ((AM.src[attribute] / AM.src["outDegree"]) * (spreading_factor))).otherwise(
                ((1 / AM.dst["inDegree"]) * AM.dst[attribute]) + (
                    (AM.src[attribute] / AM.src["outDegree"]) * (spreading_factor)))
        if transfer_function == "weighted":
            weight = AM.edge["weight"] / AM.src["w_outDegree"]
            msgToSrc = (AM.src[attribute] / AM.src["outDegree"]) * (1 - spreading_factor)
            msgToDst = sqlfunctions.when(AM.dst["outDegree"] != 0,
                                         ((AM.src[attribute]) * (spreading_factor * weight))).otherwise(
                ((1 / AM.dst["inDegree"]) * AM.dst[attribute]) + ((AM.src[attribute]) * (spreading_factor * weight)))

        # Aggregate messages
        agg = graph.aggregateMessages(sqlsum(AM.msg).alias(attribute),
                                      sendToSrc=msgToSrc,
                                      sendToDst=msgToDst)

        # Create a new cached copy of the dataFrame to get new calculated attribute
        cachedNewVertices = AM.getCachedDataFrame(agg)
        tojoin = graph.vertices.select("id", "inDegree", "outDegree", "w_inDegree", "w_outDegree")
        new_cachedNewVertices = cachedNewVertices.join(tojoin, "id", "left_outer")
        new_cachedNewVertices = new_cachedNewVertices.na.fill(0)

        # Return graph with new calculated attribute
        return GraphFrame(new_cachedNewVertices, graph.edges)

    def spread_activation_full(self, steps=0):
        """
        Full implementation of the spread activation model.
        :param steps: int, number of iterations/cycles influence is spread, if 0, accuracy_threshold used
        :return: graphframe object, new network with final calculation of attribute in vertices
        """

        # Compute degrees
        self.compute_degrees(self.graph)

        graph = self.graph

        # Number of iterations specified for spread activation
        for s in range(0, steps, 1):
            graph = self.spread_activation_step(graph, self.attribute, self.spreading_factor, self.transfer_function)

        # Number of iterations NOT specified for spread activation, use termination conditions:
        # 1) if no new nodes is infected
        # 2) if influence transferred not greater than accuracy threshold
        # WARNING: termination not guaranteed for accuracy threshold
        if steps == 0:
            max_diff = 1
            delta_infected = 1
            count = 0

            while delta_infected != 0 or max_diff > self.accuracy_threshold:
                # Iterate one spread activation
                new_graph = self.spread_activation_step(graph, self.attribute, self.spreading_factor,
                                                        self.transfer_function)

                # Compute number of infected nodes from new graph to old graph to update delta_infected
                delta_infected = len(
                    ((graph.vertices).where(graph.vertices[self.attribute] == 0)).select("id").collect()) - len(
                    ((new_graph.vertices).where(new_graph.vertices[self.attribute] == 0)).select("id").collect())

                # Compute influence difference of each node from old to new graph to update max_diff
                old_graph_v = (graph.vertices).selectExpr("id as id", self.attribute + " as old")
                new_graph_v = (new_graph.vertices).selectExpr("id as id", self.attribute + " as new")
                compare = new_graph_v.join(old_graph_v, "id", "left_outer")
                compare = compare.withColumn("diff", abs(compare["churn_new"] - compare["old"]))
                max_diff = compare.agg({"diff": "max"}).collect()[0]["max(diff)"]

                graph = new_graph
                count += 1

        # Return graph with updated attributed
        return graph