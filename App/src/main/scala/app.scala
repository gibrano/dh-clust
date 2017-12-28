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
    
    
    
    val splits = tf.flatMap(line => line.split(" ")).map(word =>(word,1))
    val counts = splits.reduceByKey((x,y)=>x+y)
    System.out.println(counts.collect().mkString(", "))
    sc.stop()
  }
}
