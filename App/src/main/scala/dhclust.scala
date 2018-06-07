package dhclust

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Clusters extends Serializable {

  def Hierarchical(layers: Array[Array[Array[Double]]], sc: SparkContext): Array[Array[Double]] = {
    var C = layers
    var linkages = Array[Array[Double]]()
    var l = C.size
    var m = C(0).size
    
    var A = layers(0)
    for(i <- 1 to (l-1)){
       A = Graph.aggregate(A,C(i))
    }
    
    val hA = Entropy.VonNewmann(A)
        
    var globalquality = Entropy.GlobalQuality(C, hA)
    var q = Array[Double](globalquality)

    while(C.size > 1){
      var n = C.size
      println("Layers size", n)
      var coords = Array[Array[Int]](Array[Int]())
      for( i <- 0 to n-2){
        for(j <- i+1 to n-1){
          coords = coords ++ Array(Array(i,j))
        }
      }
      coords = coords.filter(_.size > 0)
      var t2 = System.nanoTime
      var jsdMatrix = sc.parallelize(coords).map(x => Divergence.computeJSD(x, C))
      var duration2 = (System.nanoTime - t2) / 1e9d
      println("Duration time div JS:",duration2)
      var minimum = jsdMatrix.zipWithIndex.min
      var a = coords(minimum._2.toInt)(0)
      var b = coords(minimum._2.toInt)(1)
      var Cx = C(a)
      var Cy = C(b)
      var newlayer = Graph.aggregate(Cx,Cy)
      C = C.filter(_ != Cx)
      C = C.filter(_ != Cy)
      C = C ++ Array(newlayer)
      globalquality = Entropy.GlobalQuality(C, hA)
      println("Global quality:",globalquality)
      q = q ++ Array(globalquality)
      linkages = linkages ++ Array(Array(a,b,globalquality))
    }
    return linkages
  }
}
