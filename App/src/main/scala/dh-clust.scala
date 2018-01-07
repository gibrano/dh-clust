package dh-clust

object Clusters {
  
  def Hierarchical(C: Array[Array[org.apache.spark.mllib.linalg.Vector]]): org.apache.spark.rdd.RDD[Array[Int]] = {
    var layers = C
    var n = layers.size - 1
    var A = layers(0)
    for(i <- 1 to n){
       A = Graph.aggregate(A,layers(i))
    }
    val hA = Entropy.VonNewmann(A)
    
    var q = Array[Double]()
    var clusters = sc.parallelize(0 to n).map(i => Array(i))
    var aux = clusters
    var globalquality = Entropy.GlobalQuality(layers, hA)
    println(globalquality)
    var max = globalquality
    
    while(layers.size > 1){
       var n = layers.size
       var coords = Array[Array[Int]](Array[Int]())
       for( i <- 0 to n-2){
          for(j <- i+1 to n-1){
             coords = coords ++ Array(Array(i,j))
          }
       }
       coords = coords.filter(_.size > 0)

       var jsdMatrix = coords.map(x => Divergence.computeJSD(x, layers))
       var minimum = jsdMatrix.zipWithIndex.min
       var a = coords(minimum._2)(0)
       var b = coords(minimum._2)(1)
       var Cx = layers(a)
       var Cy = layers(b)
       var newlayer = Graph.aggregate(Cx,Cy)
       layers = layers.filter(_ != Cx)
       layers = layers.filter(_ != Cy)
       layers = layers ++ Array(newlayer)

       aux = aux.filter(!_.containsSlice(Cx))
       aux = aux.filter(!_.containsSlice(Cy))
       aux = aux ++ sc.parallelize(Array(Array(a,b)))
 
       globalquality = Entropy.GlobalQuality(layers, hA)
       println(globalquality)
       if(globalquality > max){
         max = globalquality
         clusters = aux
       }
       q = q ++ Array(globalquality)
    }

    return clusters
  }

}
