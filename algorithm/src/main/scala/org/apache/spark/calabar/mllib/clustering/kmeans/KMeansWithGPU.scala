package org.apache.spark.calabar.mllib.clustering.kmeans

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.calabar.mllib.util.MLUtils
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.BLAS.{axpy, scal}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

import scala.collection.mutable.ArrayBuffer

/**
  * Created by wx on 2017/3/15.
  */
class KMeansWithGPU(  var k: Int,
                     var maxIterations: Int,
                      var runs: Int,
                      var initializationMode: String,
                     var initializationSteps: Int,
                      var epsilon: Double,
                     var seed: Long) extends KMeans with Logging {
  private var initialModel: Option[KMeansModel] = None



  def this() = this(2, 20, 1, KMeans.K_MEANS_PARALLEL, 5, 1e-4, Utils.random.nextLong())
  private[clustering] def initKMeansParallelWithGPU(data: RDD[VectorWithNorm])
  : Array[VectorWithNorm] = {
    var costs = data.map(_ => Double.PositiveInfinity)

    val seed = new XORShiftRandom(this.seed).nextInt()
   val sample = data.takeSample(false, 1, seed)
    require(sample.nonEmpty, s"No samples available from $data")

    val centers = ArrayBuffer[VectorWithNorm]()
    var newCenters = Seq(sample.head.toDense)
    centers ++= newCenters

    var step = 0
    var bcNewCentersList = ArrayBuffer[Broadcast[_]]()
    while (step < initializationSteps) {
      val bcNewCenters = data.context.broadcast(newCenters)
      bcNewCentersList += bcNewCenters
      val preCosts = costs
      costs = data.zip(preCosts).map { case (point, cost) =>
        math.min(KMeansWithGPU.pointCost(bcNewCenters.value, point), cost)
      }.persist(StorageLevel.MEMORY_AND_DISK)
      val sumCosts = costs.sum()
      bcNewCenters.unpersist(blocking = false)
      preCosts.unpersist(blocking = false)

      val chosen = data.zip(costs).mapPartitionsWithIndex { (index, pointCosts) =>
        val rand = new XORShiftRandom(seed ^ (step << 16) ^ index)
        pointCosts.filter { case (_, c) => rand.nextDouble() < 2.0 * c * k / sumCosts }.map(_._1)
      }.collect()
      newCenters = chosen.map(_.toDense)
      centers ++= newCenters
      step += 1
    }

    costs.unpersist(blocking = false)
    bcNewCentersList.foreach(_.destroy(false))

    val distinctCenters = centers.map(_.vector).distinct.map(new VectorWithNorm(_))

    if (distinctCenters.size <= k) {
      distinctCenters.toArray
    } else {

      val bcCenters = data.context.broadcast(distinctCenters)
      val countMap = data.map(KMeansWithGPU.findClosest(bcCenters.value, _)._1).countByValue()

      bcCenters.destroy(blocking = false)

      val myWeights = distinctCenters.indices.map(countMap.getOrElse(_, 0L).toDouble).toArray
      LocalKMeans.kMeansPlusPlus(0, distinctCenters.toArray, myWeights, k, 30)
    }
  }


  private def initRandomWithGPU(data: RDD[VectorWithNorm]): Array[VectorWithNorm] = {

    data.takeSample(false, k, new XORShiftRandom(this.seed).nextInt())
      .map(_.vector).distinct.map(new VectorWithNorm(_))
  }


  private[clustering] def runAlgorithmWithGpu(data: RDD[VectorWithNorm]): KMeansModel = {

    val sc = data.sparkContext

    val initStartTime = System.nanoTime()

    val centers = initialModel match {
      case Some(kMeansCenters) =>
        kMeansCenters.clusterCenters.map(new VectorWithNorm(_))
      case None =>
        if (initializationMode == KMeans.RANDOM) {
          initRandomWithGPU(data)
        } else {
          initKMeansParallelWithGPU(data)
        }
    }
    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    logInfo(f"Initialization with $initializationMode took $initTimeInSeconds%.3f seconds.")

    var converged = false
    var iteration = 0

    val iterationStartTime = System.nanoTime()

    val start=System.currentTimeMillis()
    val rdd= data.mapPartitions(iter=>{
      val array=iter.toArray
      val vectors=array.map(_.vector)
      val pointArray=array.flatMap(_.vector.toArray.map(_.toFloat))
      val pointNorm=array.map(_.norm.toFloat)
      val kernel=new KMeansClKernel(pointArray,pointNorm,array.head.vector.size,k)
      Seq((kernel,vectors)).iterator
    }).cache()
    data.unpersist()
    while (iteration < maxIterations && !converged) {
      val bcCenters = sc.broadcast(centers)

      val totalContribs = rdd.mapPartitions{ x =>
        val thisCenters = bcCenters.value
        val value=x.next()
        val kernel=value._1
        val points=value._2
       val outputIndex= kernel.run(thisCenters.flatMap(_.vector.toArray.map(_.toFloat)),thisCenters.map(_.norm.toFloat))
        val iter= outputIndex.indices.map(x => {
          (outputIndex(x), (points(x), 1))
        }).iterator
        iter
      }.reduceByKey { case ((sum1, count1), (sum2, count2)) =>
        val sum=Vectors.dense(sum1.toArray.clone())
        axpy(1.0, sum2, sum)
        (sum, count1 + count2)
      }.collectAsMap()

      bcCenters.destroy(blocking = false)

      converged = true
      totalContribs.foreach { case (j, (sum, count)) =>

        scal(1.0 / count, sum)
        val newCenter = new VectorWithNorm(sum)
        if (converged && KMeansWithGPU.fastSquaredDistance(newCenter, centers(j)) > epsilon * epsilon) {
          converged = false
        }
        centers(j) = newCenter
      }
      iteration += 1
    }
    rdd.foreachPartition(x=>
      x.next()._1.clear()
    )
    val end=System.currentTimeMillis()
    println(s"total${end-start}")

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logInfo(f"Iterations took $iterationTimeInSeconds%.3f seconds.")

    if (iteration == maxIterations) {
      logInfo(s"KMeans.cl reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"KMeans.cl converged in $iteration iterations.")
    }
    new KMeansModel(centers.map(_.vector))
  }
  def runWithGPU(data: RDD[Vector]): KMeansModel = {

    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Compute squared norms and cache them.
    val norms = data.map(Vectors.norm(_, 2.0))
    norms.persist()
    val zippedData = data.zip(norms).map { case (v, norm) =>
      new VectorWithNorm(v)
    }

    val model = runAlgorithmWithGpu(zippedData)
    norms.unpersist()

    // Warn at the end of the run as well, for increased visibility.
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }
    model
  }

}

object   KMeansWithGPU {

  private[mllib] def pointCost(
                                centers: TraversableOnce[VectorWithNorm],
                                point: VectorWithNorm): Double =
    findClosest(centers, point)._2
  def findClosest(
                   centers: TraversableOnce[VectorWithNorm],
                   point: VectorWithNorm): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      var lowerBoundOfSqDist = center.norm - point.norm
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance: Double = fastSquaredDistance(center, point)
        if (distance < bestDistance) {
          bestDistance = distance
          bestIndex = i
        }
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }
   private[clustering]    def fastSquaredDistance(
                                               v1: VectorWithNorm,
                                               v2: VectorWithNorm): Double = {
    MLUtils.fastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
  }

  def trainWithGPU(
                    data: RDD[Vector],
                    k: Int,
                    maxIterations: Int,
                    runs: Int,
                    initializationMode: String,
                    seed: Long
                  ): KMeansModel = {
    val model = new KMeansWithGPU()
    model.k = k
    model.epsilon=runs
    model.initializationMode = initializationMode
    model.maxIterations = maxIterations
    model.runWithGPU(data)
  }

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local", "kmeans")

    val data = for (x <- 0 until math.pow(2, 20).toInt) yield {
      math.abs(x)
    }
    val rdd = sc.parallelize(data).map(x=>{
      val arr=0.to(49).map(y=>(x+y).toDouble).toArray
    Vectors.dense(arr)
    })

    rdd.cache()




    //    val data=  0 until  math.pow(2d,21d).toInt   map(x=>{
//      new VectorWithNorm(Vectors.dense(math.abs(Utils.random.nextLong()) % 10))
//    })
//
//    val centers= (0 to 127).map(x=>{
//      new VectorWithNorm(Vectors.dense(x))
//    })
//    val startGpu=System.currentTimeMillis()
//    testGPUPartitions(data.toArray ,centers.toArray)
//    val totalGPU=System.currentTimeMillis()-startGpu
//    val startCPU=System.currentTimeMillis()
//    testCpuPartition(data.toArray ,centers.toArray)
//    val totalCPU=System.currentTimeMillis()-startCPU
//    println(s"gpu算法一次迭代 kernel函数总共耗时 ${KMeansClKernel.totalTime}")
//    println(s"gpu算法一次迭代总共耗时 $totalGPU")
//    println(s"cpu一次迭代总共耗时 $totalCPU")



  }





  def testGPUPartitions(data:Array[VectorWithNorm],centers:Array[VectorWithNorm])={



  }




}

private[clustering]
class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {

  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

  def this(array: Array[Double]) = this(Vectors.dense(array))

  /** Converts the vector to a dense vector. */
  def toDense: VectorWithNorm = new VectorWithNorm(Vectors.dense(vector.toArray), norm)
}




