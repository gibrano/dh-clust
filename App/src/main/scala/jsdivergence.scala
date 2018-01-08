package dhclust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors

object Divergence {

  def JensenShannon(A: Array[org.apache.spark.mllib.linalg.Vector], B: Array[org.apache.spark.mllib.linalg.Vector], sc: SparkContext): Double = {
     var n = A.size - 1
     var C = Array[org.apache.spark.mllib.linalg.Vector]()
     for(i <- 0 to n){
        var x = Vectors.zeros(n+1)
        for(j <- 0 to n){
           x.toArray(j) = 0.5*(A(i).toArray(j) - B(i).toArray(j))
        }
        C = C ++ Array(x)
     }
     var r = Entropy.VonNewmann(C, sc)-(1/2)*(Entropy.VonNewmann(A, sc)+Entropy.VonNewmann(B, sc))
     return r
  }

  def computeJSD(x: Array[Int], layers: Array[Array[org.apache.spark.mllib.linalg.Vector]], sc: SparkContext) : Double = {
    var jsd = JensenShannon(layers(x(0)),layers(x(1)), sc)
    return jsd
  }

}
