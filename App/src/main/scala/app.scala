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

    val filename = args(0)
    val tweets = sc.textFile("s3n://AKIAJ3WA6NVC2KBLWPKQ:wBcSpSmfm1uYy1mrvUfRU0m+JyXK3O0FcAMFZyjc@gibran-bucket/tweets/"+filename)
    val texts = tweets.collect
    
    val t1 = System.nanoTime
    val tdm = TM.termDocumentMatrix(texts, sc)
    var layers = tdm.map(doc => Graph.adjacencyMatrix(doc))
    val clusters = Clusters.Hierarchical(layers, sc)
    val duration = (System.nanoTime - t1) / 1e9d
    print("Duration Time:",duration, "Numbers of Cores", sc.getExecutorStorageStatus.length)

    sc.stop()
  }
}
