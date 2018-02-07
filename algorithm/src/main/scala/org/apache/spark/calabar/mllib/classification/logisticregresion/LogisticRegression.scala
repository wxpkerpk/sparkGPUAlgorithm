/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.calabar.mllib.classification.logisticregresion

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.BLAS.dot
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.pmml.PMMLExportable
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.{DataValidators, Loader, Saveable}
import org.apache.spark.rdd.RDD

import scala.concurrent.Future
import scala.reflect.macros.Context
import scala.util.Random

class LogisticRegressionModel @Since("1.3.0")(
                                               @Since("1.0.0") override val weights: Vector,
                                               @Since("1.0.0") override val intercept: Double,
                                               @Since("1.3.0") val numFeatures: Int,
                                               @Since("1.3.0") val numClasses: Int)
  extends GeneralizedLinearModel(weights, intercept) with Serializable
    with Saveable with PMMLExportable {

  if (numClasses == 2) {
    require(weights.size == numFeatures)
  } else {
    val weightsSizeWithoutIntercept = (numClasses - 1) * numFeatures
    val weightsSizeWithIntercept = (numClasses - 1) * (numFeatures + 1)
    require(weights.size == weightsSizeWithoutIntercept || weights.size == weightsSizeWithIntercept,
      s"LogisticRegressionModel.load with numClasses = $numClasses and numFeatures = $numFeatures" +
        s" expected weights of length $weightsSizeWithoutIntercept (without intercept)" +
        s" or $weightsSizeWithIntercept (with intercept)," +
        s" but was given weights of length ${weights.size}")
  }

  private val dataWithBiasSize: Int = weights.size / (numClasses - 1)

  private val weightsArray: Array[Double] = weights match {
    case dv: DenseVector => dv.values
    case _ => null
  }


  def this(weights: Vector, intercept: Double) = this(weights, intercept, weights.size, 2)

  private var threshold: Option[Double] = Some(0.5)


  def setThreshold(threshold: Double): this.type = {
    this.threshold = Some(threshold)
    this
  }


  def getThreshold: Option[Double] = threshold


  def clearThreshold(): this.type = {
    threshold = None
    this
  }

  override protected def predictPoint(
                                       dataMatrix: Vector,
                                       weightMatrix: Vector,
                                       intercept: Double) = {
    require(dataMatrix.size == numFeatures)

    // If dataMatrix and weightMatrix have the same dimension, it's binary logistic regression.
    if (numClasses == 2) {
      val margin = dot(weightMatrix, dataMatrix) + intercept
      val score = 1.0 / (1.0 + math.exp(-margin))
      threshold match {
        case Some(t) => if (score > t) 1.0 else 0.0
        case None => score
      }
    } else {

      var bestClass = 0
      var maxMargin = 0.0
      val withBias = dataMatrix.size + 1 == dataWithBiasSize
      (0 until numClasses - 1).foreach { i =>
        var margin = 0.0
        dataMatrix.foreachActive { (index, value) =>
          if (value != 0.0) margin += value * weightsArray((i * dataWithBiasSize) + index)
        }
        // Intercept is required to be added into margin.
        if (withBias) {
          margin += weightsArray((i * dataWithBiasSize) + dataMatrix.size)
        }
        if (margin > maxMargin) {
          maxMargin = margin
          bestClass = i + 1
        }
      }
      bestClass.toDouble
    }
  }


  override protected def formatVersion: String = "1.0"

  override def toString: String = {
    s"${super.toString}, numClasses = ${numClasses}, threshold = ${threshold.getOrElse("None")}"
  }

  override def save(sc: SparkContext, path: String): Unit = {}
}


object LogisticRegressionModel {


}


class LogisticRegressionWithSGDOnGPU(
                                      private var stepSize: Double,
                                      private var numIterations: Int,
                                      private var regParam: Double,
                                      private var miniBatchFraction: Double)
  extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {
  override def generateInitialWeights(input: RDD[LabeledPoint]): Vector = {
    if (numFeatures < 0) {
      numFeatures = input.map(_.features.size).first()
    }
    if (numOfLinearPredictor == 1) {
      Vectors.zeros(numFeatures)
    } else if (addIntercept) {
      Vectors.zeros((numFeatures + 1) * numOfLinearPredictor)
    } else {
      Vectors.zeros(numFeatures * numOfLinearPredictor)
    }
  }


  def optimize(data: Array[Float], weight: Array[Float], values: Array[Float], dim: Int, stepSize: Float): Array[Float] = {
    //对于每个点,计算每一个权重的梯度
    require(dim == weight.length, "权重维度必须等于向量维度")

    def sigmod(x: Float) = 1 / (1 + math.exp(x))

    val len = data.length / dim
    val result = Array.ofDim[Float](dim)
    for (i <- 0 until len) {
      var sum = 0f
      for (j <- 0 until dim) {
        sum += data(i * dim + j) * weight(j)
      }
      //这儿在gpu上需要原子操作
      for (j <- 0 until dim) {
        val a = (sigmod(-sum) - values(i)) * data(i * dim + j) * stepSize

        result(j) += a.toFloat
      }
    }
    result

  }


  override def run(input: RDD[LabeledPoint]): LogisticRegressionModel = {


    var weight = generateInitialWeights(input).toArray.map(_.toFloat)

    def func(x: Int) = x

    (1 to 4).map(func).sum

    val rdd = input.mapPartitions {
      iter =>
        val inputVector = iter.toArray
        val inputArray = inputVector.flatMap(_.features.toArray.map(_.toFloat))
        val values = inputVector.map(_.label.toFloat)
        val logisticRegressionClKernel = new LogisticRegressionClKernel(inputArray, values, inputVector.head.features.size)
        Array(logisticRegressionClKernel).iterator
    }.cache()

    val sc = rdd.sparkContext
    0.until(numIterations).foreach {
      _ =>

        val broadcast = sc.broadcast(weight)
        val newWeight = rdd.mapPartitions { iter =>
          val weight = broadcast.value
          val kernel = iter.next()
          val result = kernel.run(weight)
          result.indices.map(x => {
            (x, result(x))
          }).iterator
        }.reduceByKey(_ + _).collectAsMap

        weight = weight.indices.map(x => {
          weight(x) - (stepSize * (newWeight(x) + regParam * weight(x))).toFloat
        }).toArray
        broadcast.destroy(false)
    }
    rdd.foreachPartition {
      //清空显存
      x => x.next.clear
    }
    rdd.unpersist(false)

    createModel(Vectors.dense(weight.map(_.toDouble)), 0)
  }


  override protected val validators = List(DataValidators.binaryLabelValidator)


  def this() = this(1.0, 100, 0.01, 1.0)

  override protected[mllib] def createModel(weights: Vector, intercept: Double) = {
    new LogisticRegressionModel(weights, intercept)
  }

  override def optimizer: Optimizer = null
}


object LogisticRegressionWithSGDOnGPU {

  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             miniBatchFraction: Double,
             initialWeights: Vector): LogisticRegressionModel = {
    new LogisticRegressionWithSGDOnGPU(stepSize, numIterations, 0.0, miniBatchFraction)
      .run(input, initialWeights)
  }


  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double,
             miniBatchFraction: Double): LogisticRegressionModel = {
    new LogisticRegressionWithSGDOnGPU(stepSize, numIterations, 0.0, miniBatchFraction)
      .run(input)
  }

  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             stepSize: Double): LogisticRegressionModel = {
    train(input, numIterations, stepSize, 1.0)
  }

  def train(
             input: RDD[LabeledPoint],
             numIterations: Int): LogisticRegressionModel = {
    train(input, numIterations, 1.0, 1.0)
  }

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext("local", "logistic")
    val len = math.pow(2, 18).toInt

    val rr = new Random(0)
    val array = (0 until len).map(_ => {
      LabeledPoint(rr.nextInt() % 2, Vectors.dense(Array(rr.nextDouble(), rr.nextDouble())))

    })
    val data = sc.parallelize(array, 2)

    val m = train(data, 100)
    m.intercept


  }

}
