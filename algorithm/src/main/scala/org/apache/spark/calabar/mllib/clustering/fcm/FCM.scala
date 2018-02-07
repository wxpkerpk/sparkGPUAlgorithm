package org.apache.spark.calabar.mllib.clustering.fcm

import org.apache.spark.SparkContext
import org.apache.spark.calabar.mllib.clustering.kmeans.{KMeansWithGPU, VectorWithNorm}
import org.apache.spark.calabar.mllib.util.Loader
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.sql.SparkSession
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.{compact, render}

/**
  * Created by wx on 2017/5 /8.
  */
class FCMWithGPU extends Serializable {


  def initcluster(data: RDD[Vector], k: Int, seed: Long = System.currentTimeMillis) = {
    data.takeSample(withReplacement = false, k, seed)
  }


  import FCMWithGPU.distance
  import FCMWithGPU.xpy
  import FCMWithGPU.ax

  import FCMWithGPU.axpby

  def run(data: RDD[Vector], k: Int, maxIterations: Int = 100, elison: Double = 0, seed: Long) = {
    var clusters = initcluster(data, k, seed)
    val sc = data.sparkContext
    var continue = true
    var iteration = 0
    val rdd = data.mapPartitions(iter => {
      val array = iter.toArray
      val arrayDouble = array.flatMap(_.toArray)
      val dataArray = arrayDouble.map(_.toFloat)
      Array((dataArray, array)).iterator

    }).persist()
    while (iteration < maxIterations && continue) {
      continue = false
      val broadcastClusters = sc.broadcast(clusters)
      val previousClusters = clusters
      val totalClusters = rdd.mapPartitions { iter => {
        val value = iter.next()
        val pointsArray = value._1
        val points = value._2
        val cluster = broadcastClusters.value
        val dim = cluster.head.size
        val resultArray = FCMClKernel.getInstance().run(pointsArray, cluster.flatMap(_.toArray.map(_.toFloat)), dim, cluster.length)

        val clusterLen = cluster.length
        val newCluster = new Array[(Double, Vector)](clusterLen).map(_ => (0d, Vectors.zeros(dim)))
        for (j <- points.indices; i <- cluster.indices) {
          val value = resultArray(i + j * clusterLen)
          (value + newCluster(i)._1, axpby(newCluster(i)._2, 1, points(j), value))
          newCluster(i) = (value + newCluster(i)._1, axpby(newCluster(i)._2, 1, points(j), value))

        }
        newCluster.indices.map(x => (x, (newCluster(x)._1, newCluster(x)._2))).iterator

      }
      }.reduceByKey { case ((u1, point1), (u2, point2)) =>

        val point = xpy(point1, point2)
        val result = (u1 + u2, point)
        result
      }.mapValues { case (u, point) =>
        ax(point, 1 / u)
      }.collectAsMap
      val end = System.currentTimeMillis()
      previousClusters.indices.foreach(x => {
        val dis = KMeansWithGPU.fastSquaredDistance(new VectorWithNorm(previousClusters(x)), new VectorWithNorm(totalClusters(x)))
        if (dis > elison * elison) continue = true
      })
      clusters = totalClusters.values.toArray
      broadcastClusters.destroy(false)
      iteration += 1

    }

    new FCMModel(clusters = clusters)
  }
}

class FCMModel(clusters: Array[Vector]) extends Saveable with Serializable {
  def predict(data: RDD[Vector]) = {
    import FCMWithGPU.caculateU
    val sc = data.sparkContext
    val broadcast = sc.broadcast(clusters)
    val result = data.map { v =>
      val clusters = broadcast.value
      val us = caculateU(v, clusters)
      (v, us)
    }
    result
  }

  def getClusters() = clusters

  override def save(sc: SparkContext, path: String): Unit = FCMModel.save(sc,this,path)

  override protected def formatVersion: String = "1.0"
}


object FCMModel extends Loader[FCMModel]
{
  def save(sc: SparkContext, model: FCMModel, path: String): Unit = {
    sc.parallelize(Seq(model.getClusters()), 1).saveAsTextFile(Loader.dataPath(path))

  }

  def load(sc: SparkContext, path: String): FCMModel = {
    implicit val formats = DefaultFormats
    val spark = SparkSession.builder().sparkContext(sc).getOrCreate()

    val centroids = spark.read.parquet(Loader.dataPath(path))
    val localCentroids = centroids.rdd.map(row=>row.getAs[Vector](0)).collect()
    new FCMModel(localCentroids)
  }
}

class FCMWithCPU extends Serializable {

  def initcluster(data: RDD[Vector], k: Int, seed: Long = System.currentTimeMillis) = {
    data.takeSample(withReplacement = false, k, seed)
  }

  import FCMWithGPU.caculateU
  import FCMWithGPU.xpy
  import FCMWithGPU.ax


  def run(data: RDD[Vector], k: Int, maxIterations: Int = 100, elison: Double = 0, seed: Long) = {
    var clusters = initcluster(data, k, seed)

    val sc = data.sparkContext
    var continue = true
    var iteration = 0

    while (iteration < maxIterations && continue) {
      continue = false
      val broadcastClusters = sc.broadcast(clusters)
      val previousClusters = clusters
      val start = System.currentTimeMillis()
      val totalClusters = data.map { v => {
        val clusters = broadcastClusters.value
        val result = caculateU(v, clusters)
        result.indices.map(x => {
          val value = result(x) * result(x)
          (x, (value, ax(v, value)))
        }).toArray

      }
      }.flatMap(_.iterator).reduceByKey { case ((u1, point1), (u2, point2)) =>

        val point = xpy(point1, point2)
        val result = (u1 + u2, point)
        result
      }.mapValues { case (u, point) =>
        ax(point, 1 / u)
      }.collectAsMap
      val end = System.currentTimeMillis()
      println(end - start)
      previousClusters.indices.foreach(x => {
        val dis = KMeansWithGPU.fastSquaredDistance(new VectorWithNorm(previousClusters(x)), new VectorWithNorm(totalClusters(x)))
        if (dis > elison * elison) continue = true
      })
      clusters = totalClusters.values.toArray
      broadcastClusters.destroy(false)
      iteration += 1

    }

    new FCMModel(clusters = clusters)
  }
}


object FCMWithGPU {
  def distance(v1: Vector, v2: Vector) =math.sqrt( Vectors.sqdist(v1, v2))

  def xpy(x: Vector, y: Vector) = {

    val array = x.toArray
    val newArray = new Array[Double](array.length)
    var i = 0
    val len = newArray.length

    while (i < len) {
      newArray(i) = x(i) + y(i)
      i += 1
    }
    Vectors.dense(newArray)
  }

  def axpby(x: Vector, a: Double, y: Vector, b: Double) = {
    val array = x.toArray
    val newArray = new Array[Double](array.length)
    var i = 0
    val len = newArray.length

    while (i < len) {
      newArray(i) = x(i) * a + y(i) * b
      i += 1
    }
    Vectors.dense(newArray)
  }

  def ax(x: Vector, a: Double) = {
    val array = x.toArray
    val newArray = new Array[Double](array.length)
    var i = 0
    val len = newArray.length

    while (i < len) {
      newArray(i) = a * x(i)
      i += 1
    }
    Vectors.dense(newArray)

  }

  def trainwithGPU(data: RDD[Vector], k: Int, maxIterations: Int = 100, elison: Double = 0, seed: Long) = {
    new FCMWithGPU().run(data, k, maxIterations, elison, seed)
  }

  def trainwithCPU(data: RDD[Vector], k: Int, maxIterations: Int = 100, elison: Double = 0, seed: Long) = {
    new FCMWithCPU().run(data, k, maxIterations, elison, seed)
  }


  def caculateU(v: Vector, clusters: Array[Vector]) = {

    val alph = 0.0000001
    val us = clusters.map { x =>
      val dis = distance(v, x) + alph
      var sum = 0.0
      for (y <- clusters) {
        val value = dis / (distance(y, v) + alph)
        sum += value * value
      }
      1 / sum
    }
    us
  }


  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local", "fcm")


    val clusters = Array(Vectors.dense(1d), Vectors.dense(2d))
    val u = caculateU(Vectors.dense(1d), clusters)
    val data = for (x <- 0 until math.pow(2, 15).toInt) yield {
      math.abs(x)
    }
    val rdd = sc.parallelize(data).map(x => {
      val arr = 0.to(1).map(y => (x + y).toDouble).toArray
      Vectors.dense(arr)
    }).cache()
    val start2 = System.currentTimeMillis()
    val m2 = trainwithGPU(rdd, 100, 10, 0, 0)
    val end2 = System.currentTimeMillis()
    val start1 = System.currentTimeMillis()
    val m = trainwithCPU(rdd, 100, 10, 0, 0)
    val end1 = System.currentTimeMillis()
    m.getClusters().foreach(println)
    m2.getClusters().foreach(println)
    println(s"gpu${end2 - start2}")
    println(s"cpu${end1 - start1}")

  }
}
