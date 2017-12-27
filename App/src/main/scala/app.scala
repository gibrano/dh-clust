package dh-clust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.{Level, Logger}

object App {
  def main(args: Array[String]) {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    val file = args(0)
    val conf = new SparkConf(true).setAppName("Distributed Hierarchical clustering")
    val sc = new SparkContext(conf)

    var tweets = sc.sc.textFile(file,2)
    var corpus = sc.parallelize(tweets)
    var tokens = corpus.map(lambda raw_text: raw_text.split()).cache()   
    var local_vocab_map = tokens.flatMap(lambda token: token).distinct().zipWithIndex().collectAsMap()

    var vocab_map = sc.broadcast(local_vocab_map)
    var vocab_size = sc.broadcast(len(local_vocab_map))

    var term_document_matrix = tokens.map(Counter).map(lambda counts: {vocab_map.value[token]: float(counts[token]) for token in counts}).map(lambda index_counts: SparseVector(vocab_size.value, index_counts))

    //display(df)

    val splits = tf.flatMap(line => line.split(" ")).map(word =>(word,1))
    val counts = splits.reduceByKey((x,y)=>x+y)
    System.out.println(counts.collect().mkString(", "))
    sc.stop()
  }
}


