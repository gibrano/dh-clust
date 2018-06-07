package dhclust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Graph extends Serializable {

    def adjacencyMatrix(v: Array[Double]): Array[Array[Double]] = {
       val n = v.size
       var A = Array(Array.fill(n)(0.00))
       for( k <- 1 to (n-1)){
          A =  A ++ Array(Array.fill(n)(0.00))
       }
       for(i <- 0 to (n-2)){
          for(j <- (i+1) to (n-1)){
             if (v(i) >= 1.00 && v(j) >= 1.00){
                A(i)(j) = 1
                A(j)(i) = 1
             }
          }
       }
       return A
    }

    def aggregate(A: Array[Array[Double]], B: Array[Array[Double]]): Array[Array[Double]] = {
        var out = Array[Array[Double]]()
        var n = A.size
        for( i <- 0 to (n-1)){
           var x = Array.fill(n)(0.00)
           for(j <- 0 to (n-1)){
               x(j) = A(i)(j) + B(i)(j)
               if(x(j) > 1){
                  x(j) = 1
               }
           }
           out = out ++ Array(x)
        }
        return out
    }
}
