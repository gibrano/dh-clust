package dhclust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Divergence extends Serializable {

  def JensenShannon(A: Array[Array[Double]], B: Array[Array[Double]]): Double = {
     var C = Array[Array[Double]]()
     var n = A.size
     for(i <- 0 to (n-1)){
       var x = Array.fill(n)(0.00)
       for(j <- 0 to (n-1)){
         x(j) = 0.5*(A(i)(j)+B(i)(j))
         if(x(j) > 1){
           x(j) = 1
         }
       }
       C = C ++ Array(x)
     }
     var r = Entropy.VonNewmann(C)-(1/2)*(Entropy.VonNewmann(A)+Entropy.VonNewmann(B))
     return r
  }

  def computeJSD(x: Array[Int], layers: Array[Array[Array[Double]]]) : Double = {
    var jsd = JensenShannon(layers(x(0)),layers(x(1)))
    return jsd
  }

}
