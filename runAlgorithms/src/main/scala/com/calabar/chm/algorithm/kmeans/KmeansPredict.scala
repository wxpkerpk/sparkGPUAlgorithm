package com.calabar.chm.algorithm.kmeans

import com.calabar.chm.spark.preprocess.{NodesUtils, Preprocessor}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession
import spray.json._
import DefaultJsonProtocol._
import com.calabar.chm.commen.RunAlgorithms
import org.apache.hadoop.fs.Path
/**
  * Created by wx on 2017/4/24.
  */
class KmeansPredict {

}
object KmeansPredict{
  def main(args: Array[String]): Unit = {



    val (ds, currNodeConfs,properties) = RunAlgorithms.getDataSet(args)


     val data=  ds.rdd.map(_.toSeq).map(s=>{
      Vectors.dense(s.map(_.toString.toDouble).toArray)
    })
    import scala.collection.JavaConverters._


    val spark = SparkSession
      .builder
      .appName("SparkKMeansWithGPU")
      //        .master("local")
      .getOrCreate()
    val modelId = currNodeConfs("nodeModelId")
    val resultPath=currNodeConfs("outpath")
    val model=KMeansModel.load(spark.sparkContext,"/CHM/model/"+modelId)
    val result= model.predict(data).map(model.clusterCenters.apply)
    val modelFile = new Path(resultPath)
    modelFile.getFileSystem(spark.sparkContext.hadoopConfiguration).delete(modelFile, true)
    data.zip(result).saveAsTextFile(resultPath)
    result


    //保存模型



    spark.stop()
  }


}
