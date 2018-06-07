package dhclust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Entropy extends Serializable {

    def VonNewmann(layer: Array[Array[Double]]): Double = {
        var A = layer
        var n = layer.size
        var entropy = 0.00
        var sumall = layer.map(row => row.sum).reduce((x,y) => x+y)
        var dgr = layer.map(row => row.sum)
        if (sumall != 0){
            var c = 1.00/sumall
            var L = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int,Double]]()
            for(i <- 0 to (n-1)){
              L(i) =  scala.collection.mutable.Map[Int,Double]()
            }    
            
            for(i <- 0 to (n-1)){
                for(j <- i to (n-1)){
                   if(i == j){
                     L(i)(j) = c*(dgr(i) - A(i)(j))
                   } else {
                     L(i)(j) = -c*A(i)(j)
                     L(j)(i) = -c*A(j)(i)
                   } 
                }
            }
            
            val eigen = Decomposition.eigenValues(L)
            for(s <- eigen){
                entropy += -s*math.log(s)
            }
        }
        return entropy
    }

    def relative(layers: Array[Array[Array[Double]]]): Double = {
       var X = layers.size
       var H = layers.map(layer => VonNewmann(layer)).reduce((x,y) => x + y)
       return H/X
    }

    def GlobalQuality(layers: Array[Array[Array[Double]]], hA: Double): Double = {
       var q = 1.00 - relative(layers)/hA
       return q
    }

}
