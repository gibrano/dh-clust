package dh-clust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, CoordinateMatrix}
import org.apache.spark.mllib.linalg.{Matrix, Matrices}

object App {
  def main(args: Array[String]) {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    
    val conf = new SparkConf(true).setAppName("Distributed Hierarchical Clustering")
    val sc = new SparkContext(conf)

    val file = args(0)
    val tweets = sc.textFile(file,2)
    val texts = tweets.collect
    val tdm = TM.termDocumentMatrix(texts)
    
    var layers = tdm.map(doc => Graph.adjacencyMatrix(doc))
    val n = layers.size    

    var coords = Array[Array[Int]](Array[Int]())
    for( i <- 0 to n-2){
       for(j <- i+1 to n-1){
          coords = coords ++ Array(Array(i,j))
       }
    }
    coords = coords.filter(_.size > 0)

    var jsdMatrix = coords.map(x => Divergence.JSDMatrix(x))

    var minimum = jsdMatrix.zipWithIndex.min
    var newlayer = Graph.aggregate(layers(coords(minimum._2)(0)),layers(coords(minimum._2)(1)))
    layers = layers.filter(_ != layers(0))
    layers = layers.filter(_ != layers(1))
    layers = layers ++ Array(newlayer)


    val cpus = Runtime.getRuntime().availableProcessors

    val jsdMatrix = Divergence.JSDMatrix(layers)

    val clusters = Clusters.Hierarchical(jsdMatrix)
 
    val clustersRdd= sc.parallelize(clusters)   
    clustersRdd.saveAsTextFile("output")
    
    sc.stop()
  }
}
