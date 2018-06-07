package dhclust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Decomposition extends Serializable {
  
  def rotate(A: scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int,Double]], k1: Int, k2: Int): scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int,Double]] = {
    val n = A.size - 1
    var ajj = A(k2).getOrElse(k2,0.00)
    var aii = A(k1).getOrElse(k1,0.00)
    var aij = A(k1).getOrElse(k2,0.00)
    var w = (ajj - aii)/(2*aij)
    var t = 0.00
    if(w>=0){
      t = -w+math.sqrt(w*w+1)
    } else {
      t = -w-math.sqrt(w*w+1)
    }
    val c = 1/(math.sqrt(1+t*t))
    val s = t/(math.sqrt(1+t*t))
    for(j <- 0 to n){
      var row1 = c*A(k1).getOrElse(j,0.00) - s*A(k2).getOrElse(j,0.00)
      var row2 = s*A(k1).getOrElse(j,0.00) + c*A(k2).getOrElse(j,0.00)
      A(k1)(j) = row1
      A(k2)(j) = row2
    }  
    for(j <- 0 to n){
      var col1 = c*A(j).getOrElse(k1,0.00) - s*A(j).getOrElse(k2,0.00)
      var col2 = s*A(j).getOrElse(k1,0.00) + c*A(j).getOrElse(k2,0.00)
      A(j)(k1) = col1
      A(j)(k2) = col2
    }  
    return A
  }
  
  def pivot(A: scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int,Double]]): (Array[Int], Double) = {
    var i = 0
    var j = 1
    var max = A(i).getOrElse(j,0.00)
    val n = A.size - 1
    for( k1 <- 0 to (n-1)){
      for( k2 <- (k1+1) to n){
        if(max < math.abs(A(k1).getOrElse(k2,0.00))){
           i = k1
           j = k2
           max = math.abs(A(k1).getOrElse(k2,0.00))
        }
      }
    }
    return (Array(i,j),max)
  }

  def getEigen(D: scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int,Double]]): Array[Double] = {
    val n = D.size - 1
    var eigenvalues = Array[Double]()
    for(i <- 0 to n){
      if(D(i)(i) > 0) {
        eigenvalues = eigenvalues ++ Array(D(i)(i))
      }
    }
    return eigenvalues
  }

  def eigenValues(A: scala.collection.mutable.Map[Int, scala.collection.mutable.Map[Int,Double]]): Array[Double] = {
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
