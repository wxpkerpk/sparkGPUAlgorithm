package org.apache.spark.calabar.mllib.clustering.Canopy

import org.apache.hadoop.fs.Path
import org.apache.spark.calabar.mllib.clustering.fcm.FCMModel
import org.apache.spark.calabar.mllib.util.Loader
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{HashPartitioner, SparkContext}
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods.{compact, render}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by wx on 2017/5/3.
  */
class CanopyWithGPU {
  def deleteFile(data: RDD[_]) = {
    val sc = data.sparkContext
    data.getCheckpointFile.foreach(file => {
      val checkpointFile = new Path(file)
      try {
        checkpointFile.getFileSystem(sc.hadoopConfiguration).delete(checkpointFile, true)
      } catch {
        case e: Exception =>
      }
    }
    )
  }


  def run(data: RDD[Vector], t1: Double, t2: Double, maxIterations: Int = 10, seed: Int = new Random().nextInt(), numPartition: Int = 10) = {
    require(t2 < t1, "t2必须小于t1")
    var rdd = data.map((CanopySet(), _, true)).persist()
    val previousRdd = rdd
    var iterations = 0
    while (maxIterations > iterations && !rdd.isEmpty()) {
      val point = rdd.takeSample(withReplacement = false, 1, 0).head
      val broadcastClusters = rdd.sparkContext.broadcast(point)
      val newRdd = rdd.filter(_._3 == true).mapPartitions { iter =>
        val cluster = broadcastClusters.value
        val dataArray = iter.toArray
        if (dataArray.length > 0) {
          val result = CanopyClKernel.getInstance().run(dataArray.flatMap(_._2.toArray), cluster._2.toArray, point._2.size)
          var i = 0
          while (i < result.length) {
            if (result(i) < t1) {
              dataArray(i)._1.add(cluster._2)
              if (result(i) < t2) {
                dataArray(i) = (dataArray(i)._1, dataArray(i)._2, false)
              }
            }
            i += 1
          }
        }
        dataArray.iterator

      }
      newRdd.persist(StorageLevel.MEMORY_AND_DISK_SER)
      if (iterations > 0) rdd.unpersist()
      rdd = newRdd
      iterations += 1
    }

    val result = previousRdd.map(x => x._1.get()
      .map(y => (y, x._2))).flatMap(_.iterator)
      .groupByKey(new HashPartitioner(numPartition))

    new CanopyModel(result)
  }

}


class CanopyWithCPU {
  def deleteFile(data: RDD[_]) = {
    val sc = data.sparkContext
    data.getCheckpointFile.foreach(file => {
      val checkpointFile = new Path(file)
      try {
        checkpointFile.getFileSystem(sc.hadoopConfiguration).delete(checkpointFile, true)
      } catch {
        case e: Exception =>
      }
    }
    )
  }


  def run(data: RDD[Vector], t1: Double, t2: Double, maxIterations: Int = 10, seed: Int = new Random().nextInt(), numPartition: Int = 10) = {
    require(t2 < t1, "t2必须小于t1")
    var rdd = data.map((CanopySet(), _, true)).persist()
    val previousRdd = rdd
    var iterations = 0
    while (maxIterations > iterations && !rdd.isEmpty()) {
      val point = rdd.takeSample(withReplacement = false, 1, 0).head
      val broadcastClusters = rdd.sparkContext.broadcast(point)
      val newRdd = rdd.filter(_._3 == true).map {
        x => {
          var result = x
          val cluster = broadcastClusters.value
          val dis = Vectors.sqdist(x._2, cluster._2)
          if (dis < t1) {
            x._1.add(cluster._2)
            if (dis < t2) {
              result = (x._1, x._2, false)
            }
          }
          result
        }
      }
      newRdd.persist(StorageLevel.MEMORY_AND_DISK_SER)
      if (iterations > 0) rdd.unpersist()
      rdd = newRdd
      iterations += 1
    }

    val result = previousRdd.map(x => x._1.get()
      .map(y => (y, x._2))).flatMap(_.iterator)
      .groupByKey(new HashPartitioner(numPartition))

    new CanopyModel(result)
  }

}

class CanopyModel(canopys: RDD[(Vector, Iterable[Vector])]) extends Saveable with Serializable {

  def getCanopys() = canopys

  override def save(sc: SparkContext, path: String): Unit = CanopyModel.save(sc, this, path)

  override protected def formatVersion: String = "1.0"
}

object CanopyModel extends Loader[CanopyModel] {
  override def load(sc: SparkContext, path: String): CanopyModel = {

    implicit val formats = DefaultFormats
    val spark = SparkSession.builder().sparkContext(sc).getOrCreate()

    val centroids = spark.read.parquet(Loader.dataPath(path))
    val localCentroids = centroids.rdd.map(_.getAs[(Vector, Iterable[Vector])](0))
    new CanopyModel(localCentroids)

  }

  def save(sc: SparkContext, model: CanopyModel, path: String): Unit = {

    val spark = SparkSession.builder().sparkContext(sc).getOrCreate()

    spark.createDataFrame(model.getCanopys()).write.parquet(Loader.dataPath(path))
  }
}

class CanopySet(data: collection.mutable.ArrayBuffer[Vector]) extends Serializable {
  def add(vec: Vector) = data += vec

  def get() = data

}

object CanopySet {
  def apply(): CanopySet = {
    val data = new ArrayBuffer[Vector]()
    new CanopySet(data)
  }

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local", "test")
    val data = (0 until 20).map(x => Vectors.dense(Array(x.toDouble)))
    val rdd = sc.parallelize(data)
    val m = new CanopyWithGPU().run(rdd, 10, 2)
    val d = m.getCanopys().collect()
    d.length
  }
}
