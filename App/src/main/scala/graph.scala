package dhclust

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix

object Graph {

    def adjacencyMatrix(v: org.apache.spark.mllib.linalg.Vector): Array[org.apache.spark.mllib.linalg.Vector] = {
       val n = v.size - 1
       var A = Array(Vectors.zeros(n+1))
       for( k <- 1 to n){
          A =  A ++ Array(Vectors.zeros(n+1))
       }

       var index = Array[Int]()
       for(i <- 0 to n){
          if (v(i) >= 1){
             index = index ++ Array(i)
          }
       }

       for(i <- index){
          for(j <- index){
             if (i < j){
                A(i).toArray(j) = 1
                A(j).toArray(i) = 1
             }
          }
       }
       return A
    }

    def sumAllEntries(A: Array[org.apache.spark.mllib.linalg.Vector]): Double = {
       var sum1 = 0.00
       for( i <- A){
          for(j <- i.toArray){
              sum1 = sum1 + j
          }
       }
       return sum1
    }

    def degrees(A: Array[org.apache.spark.mllib.linalg.Vector]): org.apache.spark.mllib.linalg.Vector = {
        var n = A.size
        var out = Vectors.zeros(n)
        for( i <- A){
           for(j <- 0 to (i.size-1)){
               out.toArray(j) = out.toArray(j) + i.toArray(j)
           }
        }
       return out
    }

    def aggregate(A: Array[org.apache.spark.mllib.linalg.Vector], B: Array[org.apache.spark.mllib.linalg.Vector]): Array[org.apache.spark.mllib.linalg.Vector] = {
        var n = A.size
        var out = Array[org.apache.spark.mllib.linalg.Vector]()
        for( i <- 0 to (n-1)){
           var x = Vectors.zeros(n)
           for(j <- 0 to (n-1)){
               x.toArray(j) = A(i).toArray(j) + B(i).toArray(j)
               if(x.toArray(j) > 1){
                  x.toArray(j) = 1
               }
           }
           out = out ++ Array(x)
        }
       return out
    }

}
