package dhclust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix

object TM {

  def termDocumentMatrix(texts: Array[String], sc: SparkContext): Array[org.apache.spark.mllib.linalg.Vector] = {
    var corpus = sc.parallelize(texts)
    var tokens = corpus.map(x => x.split(" "))
    var dwords = sc.broadcast(tokens.flatMap(token => token).distinct().collect)
    val n = dwords.value.size
    var tdm = Array[org.apache.spark.mllib.linalg.Vector]()

    var documents = sc.broadcast(tokens.collect)
    for( i <- documents.value){
       var x = Vectors.zeros(n)
       for( j <- i){
         var index = dwords.value.indexOf(j)
         x.toArray(index) = 1
       }
       tdm = tdm ++ Array(x)
    }

    return tdm
  }

}
