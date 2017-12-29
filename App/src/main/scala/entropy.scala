package dh-clust

import org.apache.spark.mllib.linalg.{Matrix, Matrices}

object Entropy {

    def relative(A: Array[org.apache.spark.mllib.linalg.Vector]): Double = {
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
            val rows = sc.parallelize(L)
            val mat: RowMatrix = new RowMatrix(rows)   
            var svd = mat.computeSVD(n,false).s
            for(s <- svd.toArray){
                entropy += -s*math.log(s)
            }
        } 
        return entropy
    }

    def VonNewmann(C: Array[org.apache.spark.mllib.linalg.Vector]): Double = {
       var H = 0.00
       var n = C.size - 1
       for(i <- 0 to n){
          var A = Graph.adjacencyMatrix(C(i))
          H += relative(A)
       }
       return H/(n+1)
    }

    def GlobalQuality(C: Array[org.apache.spark.mllib.linalg.Vector], hA: Double): Double = {
       var q = 1 - VonNewmann(C)/hA
       return q
    }

}

