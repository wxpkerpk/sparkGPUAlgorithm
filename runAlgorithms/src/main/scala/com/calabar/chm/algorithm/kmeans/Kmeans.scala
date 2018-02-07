package com.calabar.chm.algorithm.kmeans

import com.calabar.chm.commen.{Param, RunAlgorithms}
import com.calabar.chm.spark.preprocess.NodesUtils
import org.apache.spark.calabar.mllib.clustering.kmeans.KMeansWithGPU
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import spray.json._

/**
  * Created by wx on 2017/4/18.
  */
object Kmeans {


  def runKmeans(args: Array[String], algorithm: (RDD[Vector], Int, Int, Int, String, Long) => KMeansModel) = {
    implicit val appName="Kmeans"

    val (ds, currNodeConfs) = RunAlgorithms.getDataSet(args)
    val data = ds.rdd.map(_.toSeq).map(s => {
      Vectors.dense(s.map(_.toString.toDouble).toArray)
    })
    val modelId = currNodeConfs("nodeModelId")

    val result = RunAlgorithms.runAlgorithm[KMeansModel](data.sparkContext, currNodeConfs, algorithm(data, currNodeConfs("k").toInt,
      currNodeConfs("maxIterations").toInt, 0,
      currNodeConfs("initializationMode"), currNodeConfs("speed").toLong))
    result._1 += Param(result._2.k, "k", "中心点个数")
    val value = result._1
    NodesUtils.updateModel(modelId, value.toJson.toString)
  }

  object KmeansWithCPU {
    final val dataCol = "value"

    def main(args: Array[String]): Unit = {
      runKmeans(args, KMeans.train)
    }
  }

  object KmeansWithGPU {
    final val dataCol = "value"
    def main(args: Array[String]): Unit = {
      runKmeans(args, KMeansWithGPU.trainWithGPU)
    }
  }

}
