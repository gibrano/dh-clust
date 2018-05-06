package dhclust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Vector,Vectors}

object Decomposition {
  
  def rotate(A: Array[org.apache.spark.mllib.linalg.Vector], k1: Int, k2: Int): Array[org.apache.spark.mllib.linalg.Vector] = {
    val n = A.size - 1
    var w = (A(k2)(k2) - A(k1)(k1))/(2*A(k1)(k2))
    var t = 0.00
    if(w>=0){
      t = -w+math.sqrt(w*w+1)
    } else {
      t = -w-math.sqrt(w*w+1)
    }
    val c = 1/(math.sqrt(1+t*t))
    val s = t/(math.sqrt(1+t*t))
    for(j <- 0 to n){
      var row1 = c*A(k1)(j) - s*A(k2)(j)
      var row2 = s*A(k1)(j) + c*A(k2)(j)
      A(k1).toArray(j) = row1
      A(k2).toArray(j) = row2
    }  
    for(j <- 0 to n){
      var col1 = c*A(j)(k1) - s*A(j)(k2)
      var col2 = s*A(j)(k1) + c*A(j)(k2)
      A(j).toArray(k1) = col1
      A(j).toArray(k2) = col2
    }  
    return A
  }
  
  def pivot(A: Array[org.apache.spark.mllib.linalg.Vector]): (Array[Int], Double) = {
    var i = 0
    var j = 1
    var max = A(i)(j)
    val n = A.size - 1
    for( k1 <- 0 to (n-1)){
      for( k2 <- (k1+1) to n){
        if(max < math.abs(A(k1)(k2))){
           i = k1
           j = k2
           max = math.abs(A(k1)(k2))
        }
      }
    }
    return (Array(i,j),max)
  }

  def getEigen(D: Array[org.apache.spark.mllib.linalg.Vector]): Array[Double] = {
    val n = D.size - 1
    var eigenvalues = Array[Double]()
    for(i <- 0 to n){
      eigenvalues = eigenvalues ++ Array(D(i)(i))
    }
    return eigenvalues
  }

  def eigenValues(A: Array[org.apache.spark.mllib.linalg.Vector]): Array[Double] = {
    var D = A
    var n = A.size
    var iter = 0
    var max = 1.00
    while(iter < n && max > 0.1){
      var x = pivot(D)
      D = rotate(D, x._1(0), x._1(1))
      max = x._2
      iter = iter + 1
    }
    var eigen = getEigen(D)
    return eigen
  }

}
