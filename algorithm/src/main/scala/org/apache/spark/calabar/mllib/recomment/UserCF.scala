package org.apache.spark.calabar.mllib.recomment

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}
import org.json4s.DefaultFormats
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.calabar.mllib.util.Loader
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.pmml.PMMLExportable
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
/**
  * Created by wx on 2017/3/30.
  */
class UserCFWithGPU {

  final val userIdCol = "userId"
  final val productIdCol = "productId"
  final val scoreCol = "score"

  def run(data: Dataset[_]): UserCFWithGPUModel = {


    //以物品id为key作笛卡尔积,得到用户-用户 评分-评分的关系,并且过滤对角线
    val rdd = data.select(userIdCol, productIdCol, scoreCol).rdd
      .map(row => {
        (row.getString(1), (row.getString(0), row.getDouble(2)))
      })
    val matrix = rdd.join(rdd).map { case (_, ((userId1, score1), (userId2, score2))) =>

      val score = score1 - score2
      ((userId1, userId2), score * score)
    }.filter(x => x._1._1 != x._1._2)


    matrix.cache()
    //对于每一种用户-用户,统计其重叠次数
    val matrix2 = matrix.map { case ((userId1, userId2), _) =>
      ((userId1, userId2), 1)
    }.reduceByKey(_ + _)
    val matrix3 = matrix.reduceByKey(_ + _)
    //调用公式sim=m/(1+dis),dis=sqrt(x)
    val matrix4 = matrix3.join(matrix2).mapPartitions { iter => {
      val dataArray = iter.toArray
      val valueArray = dataArray.map(_._2._1)
      val mArray = dataArray.map(_._2._2)
      val result = UserCfClKernel.getInstance().run(valueArray, mArray)
      dataArray.indices.map(x => (dataArray(x)._1._1, (dataArray(x)._1._2, result(x)))).iterator
    }
    }
    new UserCFWithGPUModel(matrix4, rdd)


  }
}

class UserCFWithCPU {

  final val userIdCol = "userId"
  final val productIdCol = "productId"
  final val scoreCol = "score"

  def run(data: Dataset[_]): UserCFWithGPUModel = {

    import UserCF.distance

    //以物品id为key作笛卡尔积,得到用户-用户 评分-评分的关系,并且过滤对角线
    val rdd = data.select(userIdCol, productIdCol, scoreCol).rdd
      .map(row => {
        (row.getString(1), (row.getString(0), row.getDouble(2)))
      })
    val matrix = rdd.join(rdd).map { case (_, ((userId1, score1), (userId2, score2))) =>

      val score = score1 - score2
      ((userId1, userId2), score * score)
    }.filter(x => x._1._1 != x._1._2)


    matrix.cache()
    //对于每一种用户-用户,统计其重叠次数
    val matrix2 = matrix.map { case ((userId1, userId2), _) =>
      ((userId1, userId2), 1)
    }.reduceByKey(_ + _)
    val matrix3 = matrix.reduceByKey(_ + _)
    val matrix4 = matrix3.join(matrix2).map(x => {
      (x._1._1, (x._1._2, x._2._2 / (1 + math.sqrt(x._2._1))))
    })
    new UserCFWithGPUModel(matrix4, rdd)


  }
}

class UserCFWithGPUModel(matrix: RDD[(String, (String, Double))], itemScore: RDD[(String, (String, Double))])  extends Saveable{
  def getMatrix=matrix
  def getItemScore=itemScore
  def predict(userIdData: String, k: Int) = {
    val spark = SparkSession
      .builder()
      .sparkContext(matrix.context)
      .getOrCreate()
    type joinedType = (String, ((String, Double), Int))
    val userRdd = spark.sparkContext.parallelize(Seq((userIdData, 0)))
    //找出相似度最大的k个用户的id
    val topKUser = matrix.join(userRdd).top(k)(Ordering.by[joinedType, Double](_._2._1._2)).map(x => (x._2._1._1, x._2._1._2))
    val topKUserRdd = spark.sparkContext.parallelize(topKUser)

    val rdd = itemScore.map(f = row => (row._2._1, (row._1, row._2._2)))
    val result = rdd
      .join(topKUserRdd).map(row => (row._2._1._1, row._2._2 * row._2._1._2))
      .distinct().sortBy(_._2, ascending = false)
    val userItems = rdd.join(userRdd).map(_._2._1)
    val recItems = result.subtractByKey(userItems)

    recItems
  }

  override def save(sc: SparkContext, path: String): Unit = UserCFWithGPUModel.save(sc,this,path)

  override protected def formatVersion: String = 1.0.toString
}
object UserCFWithGPUModel extends Loader[UserCFWithGPUModel]{

  def save(sc: SparkContext, model: UserCFWithGPUModel, path: String): Unit = {
    val spark = SparkSession.builder().sparkContext(sc).getOrCreate()

    val dataRDD = model.getMatrix
    val itemRDD=model.getItemScore
    spark.createDataFrame(dataRDD).write.parquet(Loader.metadataPath(path))

    spark.createDataFrame(itemRDD).write.parquet(Loader.dataPath(path))
  }

  def load(sc: SparkContext, path: String): UserCFWithGPUModel = {
    implicit val formats = DefaultFormats
    val spark = SparkSession.builder().sparkContext(sc).getOrCreate()

    val centroids = spark.read.parquet(Loader.dataPath(path))

    val items=spark.read.parquet(Loader.metadataPath(path))

    val itemRDD=items.rdd.map{
      row=>(row.getAs[String](0),row.getAs[(String,Double)](1))

    }
    val localCentroids = centroids.rdd.map(row=> {
      val value = row.getAs[(String, Double)](1)
      (row.getAs[String](0), value)
    }
    )

    new UserCFWithGPUModel(localCentroids,itemRDD)
  }
}

object UserCF {
  def distance(v1: Vector, v2: Vector) = math.sqrt(Vectors.sqdist(v1, v2))


  final val userIdCol = "userId"
  final val productIdCol = "productId"
  final val scoreCol = "score"

  def train(data: RDD[(String, String, Double)]): UserCFWithGPUModel = {

    val spark = SparkSession
      .builder()
      .sparkContext(data.context)
      .getOrCreate()

    import spark.implicits._


    val dataset = data.map { case (userId, productId, score) => (userId, productId, score) }
      .toDF(userIdCol, productIdCol, scoreCol)
    dataset.show()
    new UserCFWithGPU().run(dataset)
  }


  def main(args: Array[String]): Unit = {
   val s=  System.getProperty("java.library.path")
    val sc = new SparkContext("local", "kmeans")

    sc.textFile("hdfs://192.168.2.50:9000/中文路径/小数据1.txt").collect().foreach(println(_))
  }

}





