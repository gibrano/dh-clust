package dhclust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object TM extends Serializable {
  def termDocumentMatrix(texts: org.apache.spark.rdd.RDD[String], sc: SparkContext): Array[Array[Double]] = {
    var tokens = texts.map(x => x.split(" "))
    var dwords = sc.broadcast(tokens.flatMap(token => token).distinct().collect)
    val n = dwords.value.size
    var tdm = Array[Array[Double]]()
    var documents = sc.broadcast(tokens.collect)
    for(i <- documents.value){
      var x = Array.fill(n)(0.00)
       for(j <- i){
         var index = dwords.value.indexOf(j)
         x(index) = 1.00
       }
       tdm = tdm ++ Array(x)
    }
    return tdm
  }
}
