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

package org.apache.spark.calabar.mllib.feature

import org.apache.spark.SparkContext
import org.apache.spark.calabar.mllib.RowMatrixWithGPU._
import org.apache.spark.mllib.feature.VectorTransformer
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


class PCAWithGPU( val k: Int) {
  require(k > 0,
    s"Number of principal components must be positive but got ${k}")


  def fit(sources: RDD[Vector]): PCAModelWithGPU = {
    val numFeatures = sources.first().size
    require(k <= numFeatures,
      s"source vector size $numFeatures must be no less than k=$k")

    val mat = new RowMatrixWithGPU(sources)

    val (pc, explainedVariance) = mat.computePrincipalComponentsAndExplainedVariance(k)
    val densePC = pc match {
      case dm: DenseMatrix =>
        dm
      case sm: SparseMatrix =>

        sm.toDense
      case m =>
        throw new IllegalArgumentException("Unsupported matrix format. Expected " +
          s"SparseMatrix or DenseMatrix. Instead got: ${m.getClass}")
    }
    val denseExplainedVariance = explainedVariance match {
      case dv: DenseVector =>
        dv
      case sv: SparseVector =>
        sv.toDense
    }
    new PCAModelWithGPU(k, densePC, denseExplainedVariance)
  }


}


class PCAModelWithGPU  (
     val k: Int,
     val pc: DenseMatrix,
    val explainedVariance: DenseVector) extends VectorTransformer {

  override def transform(vector: Vector): Vector = {
    vector match {
      case dv: DenseVector =>
        pc.transpose.multiply(dv)
      case SparseVector(size, indices, values) =>
        /* SparseVector -> single row SparseMatrix */
        val sm = Matrices.sparse(size, 1, Array(0, indices.length), indices, values).transpose
        val projection = sm.multiply(pc)
        Vectors.dense(projection.values)
      case _ =>
        throw new IllegalArgumentException("Unsupported vector format. Expected " +
          s"SparseVector or DenseVector. Instead got: ${vector.getClass}")
    }
  }
}

object PCAWithGPU extends App
{


  def train(data:RDD[Vector],k:Int)={
    new PCAWithGPU(k).fit(data)
  }

    val sc = new SparkContext("local", "pca")
    val dataLen=math.pow(2d,18d).toInt
    val k=102

    val uu=new  Random(1)

   val rdd= sc.parallelize((0 until dataLen).map(x=> {
     val arra = new ArrayBuffer[Double](k)
     (0 until k).foreach(y => arra += uu.nextDouble())
     Vectors.dense(arra.toArray)
   })).cache()


    val start=System.currentTimeMillis()
    train(rdd,50)
    val end=System.currentTimeMillis()

    println(end-start)



}
