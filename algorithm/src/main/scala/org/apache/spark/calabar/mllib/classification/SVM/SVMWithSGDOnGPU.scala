package org.apache.spark.calabar.mllib.classification.SVM

import java.io.FileWriter

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.linalg.BLAS.{axpy, dot, scal}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization.{GradientDescent, HingeGradient, SquaredL2Updater}
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
  * Created by wx on 2017/5/22.
  */

class SVMWithSGDOnGPU private(
                               private var stepSize: Double,
                               private var numIterations: Int,
                               private var regParam: Double,
                               private var miniBatchFraction: Double)
  extends GeneralizedLinearAlgorithm[SVMModel] with Serializable {

  private val gradient = new HingeGradient()
  private val updater = new SquaredL2Updater()
  override val optimizer = new GradientDescent(gradient, updater)
    .setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setRegParam(regParam)
    .setMiniBatchFraction(miniBatchFraction)
  override protected val validators = List(DataValidators.binaryLabelValidator)


  override def run(input: RDD[LabeledPoint]): SVMModel = {


    val rdd = input.mapPartitions(iter => {
      val dataArray = iter.toArray
      val features = dataArray.flatMap(_.features.toArray.map(_.toFloat))
      val values = dataArray.map(_.label.toFloat)
      val kernel = new SVMClKernel(features, values, features.length / values.length)
      Array(kernel).iterator
    }).cache()
    var weight = generateInitialWeights(input).toArray.map(_.toFloat)


    var iteration = 0
    val sc = rdd.sparkContext

    while (iteration < numIterations) {

      val broadcast = sc.broadcast(weight)

      val newWeights = rdd.mapPartitions {
        iter =>
          val kernel = iter.next
          val weight = broadcast.value
          val gradient = kernel.run(weight)
          val gradientVector = Vectors.dense(gradient.map(_.toDouble))
          scal(stepSize, gradientVector)
          Array(gradientVector).iterator
      }.reduce { case (v1, v2) =>
        axpy(1.0, v1, v2)
        v2
      }
      iteration += 1
      weight = newWeights.toArray.map(_.toFloat)
    }
    rdd.foreachPartition {
      //清空显存
      x => x.next.clear
    }
    rdd.unpersist(false)
    createModel(Vectors.dense(weight.map(_.toDouble)), 0d)

  }

  def optimize(features: Array[Float], weights: Array[Float], values: Array[Float]) = {
    val dim = weights.length
    val len = features.length / dim

    val gradient = new Array[Float](dim)


    for (i <- 0 until len) {
      val value: Float = 2 * values(i) - 1.0f
      var dotProduct = 0.0
      for (j <- 0 until dim) {
        dotProduct += features(i * dim + j) * weights(j)
      }
      if (1.0 > value * dotProduct) {
        for (k <- dim until dim) {
          gradient(k) += features(k + dim * i) * -value
        }
      }
    }
    gradient
  }

  def this() = this(1.0, 100, 0.01, 1.0)

  def compute(
               data: Vector,
               label: Double,
               weights: Vector,
               cumGradient: Vector): Double = {
    val dotProduct = dot(data, weights)

    val labelScaled = 2 * label - 1.0
    if (1.0 > labelScaled * dotProduct) {
      axpy(-labelScaled, data, cumGradient)
      1.0 - labelScaled * dotProduct
    } else {
      0.0
    }
  }

  override protected def createModel(weights: Vector, intercept: Double) = {
    new SVMModel(weights, intercept)
  }
}

object SVMWithSGDOnGPU {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local", "svm")
    val len = math.pow(2, 5).toInt

    val rr = new Random(0)
    val array = (0 until len).map(_ => {
      rr.nextInt(1000)+" "+rr.nextInt(1000)+" "+rr.nextInt(1000)
    })
   val rdd= sc.parallelize(array,1)
//   rdd.saveAsTextFile("hdfs://192.168.2.50:9000/trainSmallData2")
    val stringbuffer=new StringBuilder
//    val out = new FileWriter("C:\\output\\num1.txt",true)
    sc.textFile("hdfs://192.168.2.50:9000/ZY/小数据1(空格分割).txt").collect().foreach(println(_))





  }

  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             regParam: Double,
             miniBatchFraction: Double,
             initialWeights: Vector): SVMModel = {
    new SVMWithSGDOnGPU(stepSize, numIterations, regParam, miniBatchFraction)
      .run(input, initialWeights)
  }


  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             regParam: Double,
             miniBatchFraction: Double): SVMModel = {
    new SVMWithSGDOnGPU(stepSize, numIterations, regParam, miniBatchFraction).run(input)
  }


  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             regParam: Double): SVMModel = {
    train(input, numIterations, stepSize, regParam, 1.0)
  }


  def train(input: RDD[LabeledPoint], numIterations: Int): SVMModel = {
    train(input, numIterations, 1.0, 0.01, 1.0)
  }
}