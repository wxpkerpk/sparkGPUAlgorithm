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

package org.apache.spark.calabar.mllib.classification

import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.{ClassificationModel, NaiveBayes => NewNaiveBayes}
import org.apache.spark.mllib.classification.NaiveBayesModel.SaveLoadV2_0
import org.apache.spark.mllib.linalg.{BLAS, DenseMatrix, DenseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkContext, SparkException}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._


 class NaiveBayesWithGPUModel private[spark](
                                             val labels: Array[Double],
                                             val pi: Array[Double],
                                             val theta: Array[Array[Double]],
                                             val modelType: String) extends Saveable
   {

  import NaiveBayesWithGPU.{Bernoulli, Multinomial, supportedModelTypes}

  private val piVector = new DenseVector(pi)
  private val thetaMatrix = new DenseMatrix(labels.length, theta(0).length, theta.flatten, true)

  private[mllib] def this(labels: Array[Double], pi: Array[Double], theta: Array[Array[Double]]) =
    this(labels, pi, theta, NaiveBayesWithGPU.Multinomial)

  /** A Java-friendly constructor that takes three Iterable parameters. */
  private[mllib] def this(
                           labels: Iterable[Double],
                           pi: Iterable[Double],
                           theta: Iterable[Iterable[Double]]) =
    this(labels.toArray, pi.toArray, theta.toArray.map(_.toArray))

  require(supportedModelTypes.contains(modelType),
    s"Invalid modelType $modelType. Supported modelTypes are $supportedModelTypes.")

  // Bernoulli scoring requires log(condprob) if 1, log(1-condprob) if 0.
  // This precomputes log(1.0 - exp(theta)) and its sum which are used for the linear algebra
  // application of this condition (in predict function).
  private val (thetaMinusNegTheta, negThetaSum) = modelType match {
    case Multinomial => (None, None)
    case Bernoulli =>
      val negTheta = thetaMatrix.map(value => math.log(1.0 - math.exp(value)))
      val ones = new DenseVector(Array.fill(thetaMatrix.numCols) {
        1.0
      })
      val thetaMinusNegTheta = thetaMatrix.map { value =>
        value - math.log(1.0 - math.exp(value))
      }
      (Option(thetaMinusNegTheta), Option(negTheta.multiply(ones)))
    case _ =>
      // This should never happen.
      throw new UnknownError(s"Invalid modelType: $modelType.")
  }

  @Since("1.0.0")
   def predict(testData: RDD[Vector]): RDD[Double] = {
    val bcModel = testData.context.broadcast(this)
    modelType match {
      case Multinomial => {
        testData.mapPartitions { iter =>
          val model = bcModel.value
          val data = iter.toArray
          val matrixArray = model.thetaMatrix.toArray
          val piArray = model.piVector.toArray
          val numFeatures = data.head.size
          val numClass = piArray.length
          val outPut = Array.fill[Int](data.length)(2)
          NaiveBayesClKernel.getInstance().run(matrixArray, piArray, null, data.flatMap(_.toArray), numFeatures, numClass, outPut)
          data.indices.map(x => labels(outPut(x))).iterator
        }
      }
      case Bernoulli => {
        testData.mapPartitions { iter =>
          val model = bcModel.value
          val data = iter.toArray
          val matrixArray = model.thetaMinusNegTheta.get.toArray
          val piArray2 = model.negThetaSum.get.toArray
          val piArray = model.piVector.toArray
          val numFeatures = data.head.size
          val numClass = piArray.length
          val outPut = Array.fill[Int](data.length)(2)
          NaiveBayesClKernel.getInstance().run(matrixArray, piArray, piArray2, data.flatMap(_.toArray), numFeatures, numClass, outPut)
          data.indices.map(x => labels(outPut(x))).iterator
        }
      }


    }
  }


  @Since("1.0.0")
   def predict(testData: Vector): Double = {
    modelType match {
      case Multinomial =>
        labels(multinomialCalculation(testData).argmax)
      case Bernoulli =>
        labels(bernoulliCalculation(testData).argmax)
    }
  }


  def predictProbabilities(testData: RDD[Vector]): RDD[Vector] = {
    val bcModel = testData.context.broadcast(this)
    testData.mapPartitions { iter =>
      val model = bcModel.value
      iter.map(model.predictProbabilities)
    }
  }


  def predictProbabilities(testData: Vector): Vector = {
    modelType match {
      case Multinomial =>
        posteriorProbabilities(multinomialCalculation(testData))
      case Bernoulli =>
        posteriorProbabilities(bernoulliCalculation(testData))
    }
  }

  private def multinomialCalculation(testData: Vector) = {

    val prob = thetaMatrix.multiply(testData)
    BLAS.axpy(1.0, piVector, prob)
    prob
  }

  private def bernoulliCalculation(testData: Vector) = {
    testData.foreachActive((_, value) =>
      if (value != 0.0 && value != 1.0) {
        throw new SparkException(
          s"Bernoulli naive Bayes requires 0 or 1 feature values but found $testData.")
      }
    )
    val prob = thetaMinusNegTheta.get.multiply(testData)
    BLAS.axpy(1.0, piVector, prob)
    BLAS.axpy(1.0, negThetaSum.get, prob)
    prob
  }

  private def posteriorProbabilities(logProb: DenseVector) = {
    val logProbArray = logProb.toArray
    val maxLog = logProbArray.max
    val scaledProbs = logProbArray.map(lp => math.exp(lp - maxLog))
    val probSum = scaledProbs.sum
    new DenseVector(scaledProbs.map(_ / probSum))
  }

     override def save(sc: SparkContext, path: String): Unit = null

     override protected def formatVersion: String = null
   }

object NaiveBayesWithGPUModelWithGPU extends Loader[NaiveBayesWithGPUModel] {


  private[mllib] object SaveLoadV2_0 {

    def thisFormatVersion: String = "2.0"

    /** Hard-code class name string in case it changes in the future */
    def thisClassName: String = "NaiveBayesWithGPUModel"

    /** Model data for model import/export */
    case class Data(
                     labels: Array[Double],
                     pi: Array[Double],
                     theta: Array[Array[Double]],
                     modelType: String)


  }

  override def load(sc: SparkContext, path: String): NaiveBayesWithGPUModel = {
    null
  }
}


class NaiveBayesWithGPU private(
                                 private var lambda: Double,
                                 private var modelType: String) extends Serializable with Logging {

  def this(lambda: Double) = this(lambda, NaiveBayesWithGPU.Multinomial)

  def this() = this(1.0, NaiveBayesWithGPU.Multinomial)

  /** Set the smoothing parameter. Default: 1.0. */
  def setLambda(lambda: Double): NaiveBayesWithGPU = {
    require(lambda >= 0,
      s"Smoothing parameter must be nonnegative but got $lambda")
    this.lambda = lambda
    this
  }

  /** Get the smoothing parameter. */
  def getLambda: Double = lambda

  /**
    * Set the model type using a string (case-sensitive).
    * Supported options: "multinomial" (default) and "bernoulli".
    */
  def setModelType(modelType: String): NaiveBayesWithGPU = {
    require(NaiveBayesWithGPU.supportedModelTypes.contains(modelType),
      s"NaiveBayes was created with an unknown modelType: $modelType.")
    this.modelType = modelType
    this
  }

  def getModelType: String = this.modelType


  def run(data: RDD[LabeledPoint]): NaiveBayesWithGPUModel = {
    val spark = SparkSession
      .builder()
      .sparkContext(data.context)
      .getOrCreate()

    import spark.implicits._

    val nb = new NewNaiveBayes()
      .setModelType(modelType)
      .setSmoothing(lambda)

    val dataset = data.map { case LabeledPoint(label, features) => (label, features.asML) }
      .toDF("label", "features")

    // org.apache.spark.mllib NaiveBayes allows input labels like {-1, +1}, so set `positiveLabel` as false.
    val newModel = nb.trainWithLabelCheck(dataset, positiveLabel = false)

    val pi = newModel.pi.toArray
    val theta = Array.fill[Double](newModel.numClasses, newModel.numFeatures)(0.0)
    newModel.theta.foreachActive {
      case (i, j, v) =>
        theta(i)(j) = v
    }

    assert(newModel.oldLabels != null,
      "The underlying ML NaiveBayes training does not produce labels.")
    new NaiveBayesWithGPUModel(newModel.oldLabels, pi, theta, modelType)
  }
}


object NaiveBayesWithGPU {

  private[classification] val Multinomial: String = "multinomial"

  private[classification] val Bernoulli: String = "bernoulli"

  private[classification] val supportedModelTypes = Set(Multinomial, Bernoulli)


  def train(input: RDD[LabeledPoint]): NaiveBayesWithGPUModel = {
    new NaiveBayesWithGPU().run(input)
  }


  def train(input: RDD[LabeledPoint], lambda: Double): NaiveBayesWithGPUModel = {
    new NaiveBayesWithGPU(lambda, Multinomial).run(input)
  }


  def train(input: RDD[LabeledPoint], lambda: Double, modelType: String): NaiveBayesWithGPUModel = {
    require(supportedModelTypes.contains(modelType),
      s"NaiveBayes was created with an unknown modelType: $modelType.")
    new NaiveBayesWithGPU(lambda, modelType).run(input)
  }


  def cover(value:Int) ={
    if(value<100) 2 else if(value<900)  3 else 4
  }

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local", "kmeans")

    val data = for (x <- 0 until math.pow(2, 10).toInt) yield {
      x
    }
    val rdd = sc.parallelize(data, 1).map(x => new LabeledPoint(cover(x), Vectors.dense(Array(x.toDouble%2, x%2))))
    val start1 = System.currentTimeMillis()
    val m2 = train(rdd,1,NaiveBayesWithGPU.Bernoulli)

     val s=  m2.predict(sc.parallelize(Seq{Vectors.dense(Array(1.toDouble, 1))}))          .collect()
    val s2=m2.predict(Vectors.dense(Array(1.toDouble, 1)))
    val end1 = System.currentTimeMillis()

    println(end1 - start1)


  }

}
