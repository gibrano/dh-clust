package dhclust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, Matrices}

object Entropy {

    def VonNewmann(A: Array[org.apache.spark.mllib.linalg.Vector], sc: SparkContext): Double = {
        var entropy = 0.00
        var out = 0.00
        val E = Graph.sumAllEntries(A)
        if (E != 0){
            val c = 1/E
            val degr = Graph.degrees(A)
            val D = Matrices.diag(degr)
            val n = D.numRows - 1
            var L = Array[org.apache.spark.mllib.linalg.Vector]()
            for(i <- 0 to n){
                var x = Vectors.zeros(n+1)
                for(j <- 0 to n){
                   x.toArray(j) = c*(D(i,j) - A(i).toArray(j))
                }
                L = L ++ Array(x)
            }
            val eigen = Decomposition.eigenValues(L,sc)
            for(s <- eigen){
                entropy += -s*math.log(s)
            }
        }
        return entropy
    }

    def relative(layers: Array[Array[org.apache.spark.mllib.linalg.Vector]],sc: SparkContext): Double = {
       //var H = 0.00
       var n = layers.size - 1
       // for(i <- 0 to n){
       //    H += VonNewmann(layers(i), sc)
       // }
       var H = layers.map(C => VonNewmann(C, sc)).reduce((x,y) => x + y)
       return H/(n+1)
    }

    def GlobalQuality(layers: Array[Array[org.apache.spark.mllib.linalg.Vector]], hA: Double, sc: SparkContext): Double = {
       var q = 1 - relative(layers, sc)/hA
       return q
    }

}
