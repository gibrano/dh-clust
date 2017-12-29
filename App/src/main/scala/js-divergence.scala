package dh-clust

object Divergence {
  
  def JensenShannon(A: Array[org.apache.spark.mllib.linalg.Vector], B: Array[org.apache.spark.mllib.linalg.Vector]): Double = {
     var n = A.size - 1
     var C = Array[org.apache.spark.mllib.linalg.Vector]()
     for(i <- 0 to n){
        var x = Vectors.zeros(n+1)
        for(j <- 0 to n){
           x.toArray(j) = 0.5*(A(i).toArray(j) - B(i).toArray(j))
        }
        C = C ++ Array(x) 
     }
     var r = Entropy.relative(C)-(1/2)*(Entropy.relative(A)+Entropy.relative(B))
     return r
  }

  def JSDMatrix(C: Array[Array[org.apache.spark.mllib.linalg.Vector]] ) : Array[org.apache.spark.mllib.linalg.Vector] {
    
    return distances
  }
}

