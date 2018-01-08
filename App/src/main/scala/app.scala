package dhclust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.{Level, Logger}

object App {
  def main(args: Array[String]) {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val conf = new SparkConf(true).setAppName("Distributed Hierarchical Clustering")
    val sc = new SparkContext(conf)

    val file = args(0)
    val tweets = sc.textFile(file,2)
    val texts = tweets.take(20)
    val tdm = TM.termDocumentMatrix(texts, sc)

    var layers = tdm.map(doc => Graph.adjacencyMatrix(doc))

    val clusters = Clusters.Hierarchical(layers, sc)

    var clustersRDD = sc.parallelize(clusters)
    clustersRDD.saveAsTextFile("output")

    sc.stop()
  }
}
