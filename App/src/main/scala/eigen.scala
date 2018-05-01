package dhclust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Vector,Vectors}

object Decomposition {
  
  def eye(n: Int): Array[org.apache.spark.mllib.linalg.Vector] = {
    var I =  Array[org.apache.spark.mllib.linalg.Vector]()
    for( i <- 0 to (n-1)){
      var x = Vectors.zeros(n)
      for(j <- 0 to (n-1)){
        if(i == j){
           x.toArray(j) = 1.00
        }
      }
      I = I ++ Array(x)
    }
    return I
  }

  def rotation(A: Array[org.apache.spark.mllib.linalg.Vector], i: Int, j: Int): Array[org.apache.spark.mllib.linalg.Vector] = {
    var w = (A(j)(j) - A(i)(i))/(2*A(i)(j))
    var t = 0.00
    var n = A.size - 1
    if(w>=0){
      t = -w+math.sqrt(w*w+1)
    } else {
      t = -w-math.sqrt(w*w+1)
    }
    val c = 1/(math.sqrt(1+t*t))
    val s = t/(math.sqrt(1+t*t))
    var R = eye(A.size)
    R(i).toArray(i) = c
    R(i).toArray(j) = s
    R(j).toArray(i) = -s
    R(j).toArray(j) = c
    return R
  }

  def pivot(A: Array[org.apache.spark.mllib.linalg.Vector]): Array[Int] = {
    var i = 0
    var j = 1
    val n = A.size - 1
    for( k1 <- 0 to n){
      for(k2 <- 0 to n){
        if(math.abs(A(i)(j)) < math.abs(A(k1)(k2))){
           i = k1
           j = k2
        }
      }
    }
    return Array(i,j)
  }
  
  def eigenValues(A: Array[org.apache.spark.mllib.linalg.Vector]): Array[Double] = {
    var D = A
    var err = 1.00
    while(err < 0.01){
      var x = pivot(D)
      var R = rotation(D,x(1),x(2))
      var RD = prod(R, D)
      D = prod(RD, R)
      err = getMax(D)
    }
    var v = getEigen(D)
    return v
  }

}

