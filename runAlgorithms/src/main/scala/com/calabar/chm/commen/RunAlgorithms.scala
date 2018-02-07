package com.calabar.chm.commen

import com.calabar.chm.spark.preprocess.{NodesUtils, Preprocessor}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.Saveable

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by wx on 2017/6/7.
  */
object RunAlgorithms {
  type resultType = ArrayBuffer[Param]


  def runAlgorithm[T <: Saveable](sc: SparkContext, paramaters: mutable.Map[String, String], algorithm: => T): (ArrayBuffer[Param], T) = {
    val modelId = paramaters("nodeModelId")

    //生成模型描述
    val modelPath = "/CHM/model/" + modelId
    val modelFile = new Path(modelPath)
    modelFile.getFileSystem(sc.hadoopConfiguration).delete(modelFile, true)
    val start = System.currentTimeMillis()
    val model = algorithm
    model.save(sc, modelPath)
    val end = System.currentTimeMillis()
    val result = (Param(start, "start", "开始时间") :: Param(end, "end", "结束时间")) ++ (Param(modelPath, "modelPath", "模型路径") :: Param(modelId, "modelId", "模型id"))

    (result, model)


  }



  def getDataSetAndProperties(args: Array[String])(implicit appName:String)={
    val (taskDef_id, currId) = (args(0), args(1))
    import scala.collection.JavaConverters._

    val currNodeConfs = NodesUtils.getIns(taskDef_id).loadNodeConf(currId).asScala
    val spark = SparkSession
      .builder
      .appName(appName)
      //        .master("local")
      .getOrCreate()
    val start = System.currentTimeMillis()
    val end = System.currentTimeMillis()

    val modelId = currNodeConfs("nodeModelId")
    val properties= NodesUtils.loadSCNodeConf(taskDef_id,currId).asScala
    val resultPath=properties("outpath")
    val modelFile = new Path(resultPath)
    modelFile.getFileSystem(spark.sparkContext.hadoopConfiguration).delete(modelFile, true)
    ("/CHM/model/"+modelId,resultPath)

  }

  def getDataSet(args: Array[String])(implicit appName:String) = {
    //流程定义id
    import scala.collection.JavaConverters._

    val (taskDef_id, currId) = (args(0), args(1))
    //当前节点id
    val spark = SparkSession
      .builder
      .appName(appName)
      //        .master("local")
      .getOrCreate()
    NodesUtils.getIns(taskDef_id).updateAPPID(currId, spark.sparkContext.applicationId)
    val ds = Preprocessor(spark, taskDef_id).process(currId)
    val currNodeConfs = NodesUtils.getIns(taskDef_id).loadNodeConf(currId).asScala

    require(ds != null, "err:运行时出错")
    (ds, currNodeConfs)

  }
}
