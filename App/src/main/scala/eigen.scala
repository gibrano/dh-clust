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
    for( k1 <- 0 to (n-1)){
      for(k2 <- (k1+1) to n){
        if(math.abs(A(i)(j)) < math.abs(A(k1)(k2))){
           i = k1
           j = k2
        }
      }
    }
    return Array(i,j)
  }
  
  def vectorProd(x: org.apache.spark.mllib.linalg.Vector, B: Array[org.apache.spark.mllib.linalg.Vector], sc: SparkContext): org.apache.spark.mllib.linalg.Vector = {
    val n = x.size - 1
    val m = B(0).size - 1
    var sum = 0.0
    var z = Vectors.zeros(m+1)
    var index = sc.parallelize(0 to n)
    for(i <- 0 to m){
      sum = index.map(j => x(j)*B(j)(i)).reduce(_+_)
      z.toArray(i) = sum
    }  
    return z
  }
    
  def matrixProd(A: Array[org.apache.spark.mllib.linalg.Vector], B: Array[org.apache.spark.mllib.linalg.Vector], sc: SparkContext): Array[org.apache.spark.mllib.linalg.Vector] = {
    val n = A.size - 1
    val m = B(0).size - 1
    var C = Array[org.apache.spark.mllib.linalg.Vector]()
    for( i <- 0 to n){
      var x = vectorProd(A(i), B, sc)
      C = C ++ Array(x)
    }
    return C
  }

  def getError(A: Array[org.apache.spark.mllib.linalg.Vector]): Double = {
    val n = A.size - 1
    var error = math.abs(A(0)(1))
    for( i <- 0 to (n-1)){
      for(j <- (i+1) to n){
        if(error < math.abs(A(i)(j))){
           error = math.abs(A(i)(j))
        }
      }
    }
    return error
}
  
  def getEigen(D: Array[org.apache.spark.mllib.linalg.Vector]): Array[Double] = {
    val n = D.size - 1
    var values = Array[Double]()
    var eigenvalues = Array[Double]()
    for( i <- 0 to n){
      values = values ++ Array(D(i)(i))
    }
    values = values.sorted
    for( i <- 0 to n){
      eigenvalues = eigenvalues ++ Array(values(n-i))
    }
    return eigenvalues
  }
  
  def eigenValues(A: Array[org.apache.spark.mllib.linalg.Vector], sc: SparkContext): Array[Double] = {
    var D = A
    var err = 1.00
    while(err > 0.01){
      var x = pivot(D)
      var R = rotation(D,x(0),x(1))
      var Rt = rotation(D,x(0),x(1))
      Rt(x(0)).toArray(x(1)) = -Rt(x(0))(x(1))
      Rt(x(1)).toArray(x(0)) = -Rt(x(1))(x(0))
      var RtD = matrixProd(Rt, D, sc)
      D = matrixProd(RtD, R, sc)
      err = getError(D)
    }
    var v = getEigen(D)
    return v
  }

}

