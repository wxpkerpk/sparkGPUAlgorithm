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

package org.apache.spark.calabar.mllib.RowMatrixWithGPU

import java.util.Arrays

import breeze.linalg.{MatrixSingularException, inv, DenseVector => BDV, SparseVector => BSV, axpy => brzAxpy, svd => brzSvd}
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.stat.{MultivariateOnlineSummarizer, MultivariateStatisticalSummary}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom
import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, Matrix => BM}

import scala.collection.mutable.ListBuffer

/**
  * Represents a row-oriented distributed Matrix with no meaningful row indices.
  *
  * @param rows  rows stored as an RDD[Vector]
  * @param nRows number of rows. A non-positive value means unknown, and then the number of rows will
  *              be determined by the number of records in the RDD `rows`.
  * @param nCols number of columns. A non-positive value means unknown, and then the number of
  *              columns will be determined by the size of the first row.
  */
class RowMatrixWithGPU (
                                  val rows: RDD[Vector],
                                 private var nRows: Long,
                                 private var nCols: Int)extends Logging {




  def this(rows: RDD[Vector]) = this(rows, 0L, 0)

   def numCols(): Long = {
    if (nCols <= 0) {
      try {
        // Calling `first` will throw an exception if `rows` is empty.
        nCols = rows.first().size
      } catch {
        case err: UnsupportedOperationException =>
          sys.error("Cannot determine the number of cols because it is not specified in the " +
            "constructor and the rows RDD is empty.")
      }
    }
    nCols
  }


   def numRows(): Long = {
    if (nRows <= 0L) {
      nRows = rows.count()
      if (nRows == 0L) {
        sys.error("Cannot determine the number of rows because it is not specified in the " +
          "constructor and the rows RDD is empty.")
      }
    }
    nRows
  }

  implicit class covertToBreeze(matrix: Matrix) {
    def asBreezes(): BM[Double] = {
      matrix match {

        case dense: DenseMatrix =>

          if (!dense.isTransposed) {
            new BDM[Double](dense.numRows, dense.numCols, dense.values)
          } else {
            val breezeMatrix = new BDM[Double](dense.numCols, dense.numRows, dense.values)
            breezeMatrix.t
          }
        case sparse: SparseMatrix =>
          if (!sparse.isTransposed) {
            new BSM[Double](sparse.values, sparse.numRows, sparse.numCols, sparse.colPtrs, sparse.rowIndices)
          } else {
            val breezeMatrix = new BSM[Double](sparse.values, sparse.numCols, sparse.numRows, sparse.colPtrs, sparse.rowIndices)
            breezeMatrix.t
          }
        case _ => null
      }


    }
  }


     def fromBreeze(breeze: BM[Double]): Matrix = {
      breeze match {
        case dm: BDM[Double] =>
          new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)
        case sm: BSM[Double] =>
          // There is no isTranspose flag for sparse matrices in Breeze
          new SparseMatrix(sm.rows, sm.cols, sm.colPtrs, sm.rowIndices, sm.data)
        case _ =>
          throw new UnsupportedOperationException(
            s"Do not support conversion from type ${breeze.getClass.getName}.")
      }
    }
      def toBreeze(): BDM[Double] = {
      val m = numRows().toInt
      val n = numCols().toInt
      val mat = BDM.zeros[Double](m, n)
      var i = 0
      rows.collect().foreach { vector =>
        vector.foreachActive { case (j, v) =>
          mat(i, j) = v
        }
        i += 1
      }
      mat
    }
    /**
      * Multiplies the Gramian matrix `A^T A` by a dense vector on the right without computing `A^T A`.
      *
      * @param v a dense vector whose length must match the number of columns of this matrix
      * @return a dense vector representing the product
      */
     def multiplyGramianMatrixBy(v: BDV[Double]): BDV[Double] = {
      val n = numCols().toInt
      val vbr = rows.context.broadcast(v)
      rows.treeAggregate(BDV.zeros[Double](n))(
        seqOp = (U, r) => {
          val rBrz = r.asBreeze
          val a = rBrz.dot(vbr.value)
          rBrz match {
            // use specialized axpy for better performance
            case _: BDV[_] => brzAxpy(a, rBrz.asInstanceOf[BDV[Double]], U)
            case _: BSV[_] => brzAxpy(a, rBrz.asInstanceOf[BSV[Double]], U)
            case _ => throw new UnsupportedOperationException(
              s"Do not support vector operation from type ${rBrz.getClass.getName}.")
          }
          U
        }, combOp = (U1, U2) => U1 += U2)
    }

    /**
      * Computes the Gramian matrix `A^T A`.
      *
      * @note This cannot be computed on matrices with more than 65535 columns.
      */
    def computeGramianMatrix(): Matrix = {
      val n = numCols().toInt
      checkNumColumns(n)
      // Computes n*(n+1)/2, avoiding overflow in the multiplication.
      // This succeeds when n <= 65535, which is checked above
      val nt = if (n % 2 == 0) ((n / 2) * (n + 1)) else (n * ((n + 1) / 2))

      // Compute the upper triangular part of the gram matrix.
      val GU = rows.treeAggregate(new BDV[Double](nt))(
        seqOp = (U, v) => {
          BLAS.spr(1.0, v, U.data)
          U
        }, combOp = (U1, U2) => U1 += U2)

      RowMatrixWithGPU.triuToFull(n, GU.data)
    }

    def checkNumColumns(cols: Int): Unit = {
      if (cols > 65535) {
        throw new IllegalArgumentException(s"Argument with more than 65535 cols: $cols")
      }
      if (cols > 10000) {
        val memMB = (cols.toLong * cols) / 125000
        logWarning(s"$cols columns will require at least $memMB megabytes of memory!")
      }
    }


    def computeCovariance(): Matrix = {
      val n = numCols().toInt
      checkNumColumns(n)

      val summary = computeColumnSummaryStatistics()
      val m = summary.count
      require(m > 1, s"RowMatrix.computeCovariance called on matrix with only $m rows." +
        "  Cannot compute the covariance of a RowMatrix with <= 1 row.")
      val mean = summary.mean

      // We use the formula Cov(X, Y) = E[X * Y] - E[X] E[Y], which is not accurate if E[X * Y] is
      // large but Cov(X, Y) is small, but it is good for sparse computation.
      // TODO: find a fast and stable way for sparse data.

      val G = computeGramianMatrix().asBreezes()

      var i = 0
      var j = 0
      val m1 = m - 1.0
      var alpha = 0.0
      while (i < n) {
        alpha = m / m1 * mean(i)
        j = i
        while (j < n) {
          val Gij = G(i, j) / m1 - alpha * mean(j)
          G(i, j) = Gij
          G(j, i) = Gij
          j += 1
        }
        i += 1
      }

      fromBreeze(G)
    }

    def computePrincipalComponentsAndExplainedVariance(k: Int): (Matrix, Vector) = {
      val n = numCols().toInt
      require(k > 0 && k <= n, s"k = $k out of range (0, n = $n]")

      val start1=System.currentTimeMillis()
      val Cov = computeCovariance().asBreezes().asInstanceOf[BDM[Double]]
      val end1=System.currentTimeMillis()
      println(end1-start1)


      val start2=System.currentTimeMillis()
      val brzSvd.SVD(u: BDM[Double], s: BDV[Double], _) = brzSvd(Cov)
      val end2=System.currentTimeMillis()
      println(end2-start2)

      val eigenSum = s.data.sum
      val explainedVariance = s.data.map(_ / eigenSum)

      if (k == n) {
        (Matrices.dense(n, k, u.data), Vectors.dense(explainedVariance))
      } else {
        (Matrices.dense(n, k, Arrays.copyOfRange(u.data, 0, n * k)),
          Vectors.dense(Arrays.copyOfRange(explainedVariance, 0, k)))
      }
    }

    /**
      * Computes the top k principal components only.
      *
      * @param k number of top principal components.
      * @return a matrix of size n-by-k, whose columns are principal components
      * @see computePrincipalComponentsAndExplainedVariance
      */
    def computePrincipalComponents(k: Int): Matrix = {
      computePrincipalComponentsAndExplainedVariance(k)._1
    }

    /**
      * Computes column-wise summary statistics.
      */
    def computeColumnSummaryStatistics(): MultivariateStatisticalSummary = {
      val summary = rows.treeAggregate(new MultivariateOnlineSummarizer)(
        (aggregator, data) => aggregator.add(data),
        (aggregator1, aggregator2) => aggregator1.merge(aggregator2))
      updateNumRows(summary.count)
      summary
    }

    /**
      * Multiply this matrix by a local matrix on the right.
      *
      * @param B a local matrix whose number of rows must match the number of columns of this matrix
      * @return a [[org.apache.spark.mllib.linalg.distributed.RowMatrix]] representing the product,
      *         which preserves partitioning
      */
    def multiply(B: Matrix): RowMatrixWithGPU = {
      val n = numCols().toInt
      val k = B.numCols
      require(n == B.numRows, s"Dimension mismatch: $n vs ${B.numRows}")

      require(B.isInstanceOf[DenseMatrix],
        s"Only support dense matrix at this time but found ${B.getClass.getName}.")

      val Bb = rows.context.broadcast(B.asBreezes.asInstanceOf[BDM[Double]].toDenseVector.toArray)
      val AB = rows.mapPartitions { iter =>
        val Bi = Bb.value
        iter.map { row =>
          val v = BDV.zeros[Double](k)
          var i = 0
          while (i < k) {
            v(i) = row.asBreeze.dot(new BDV(Bi, i * n, 1, n))
            i += 1
          }
          Vectors.fromBreeze(v)
        }
      }

      new RowMatrixWithGPU(AB, nRows, B.numCols)
    }

    def columnSimilarities(): CoordinateMatrix = {
      columnSimilarities(0.0)
    }


    def columnSimilarities(threshold: Double): CoordinateMatrix = {
      require(threshold >= 0, s"Threshold cannot be negative: $threshold")

      if (threshold > 1) {
        logWarning(s"Threshold is greater than 1: $threshold " +
          "Computation will be more efficient with promoted sparsity, " +
          " however there is no correctness guarantee.")
      }

      val gamma = if (threshold < 1e-6) {
        Double.PositiveInfinity
      } else {
        10 * math.log(numCols()) / threshold
      }

      columnSimilaritiesDIMSUM(computeColumnSummaryStatistics().normL2.toArray, gamma)
    }

    def tallSkinnyQR(computeQ: Boolean = false): QRDecomposition[RowMatrixWithGPU, Matrix] = {
      val col = numCols().toInt
      // split rows horizontally into smaller matrices, and compute QR for each of them
      val blockQRs = rows.retag(classOf[Vector]).glom().filter(_.length != 0).map { partRows =>
        val bdm = BDM.zeros[Double](partRows.length, col)
        var i = 0
        partRows.foreach { row =>
          bdm(i, ::) := row.asBreeze.t
          i += 1
        }
        breeze.linalg.qr.reduced(bdm).r
      }

      // combine the R part from previous results vertically into a tall matrix
      val combinedR = blockQRs.treeReduce { (r1, r2) =>
        val stackedR = BDM.vertcat(r1, r2)
        breeze.linalg.qr.reduced(stackedR).r
      }

      val finalR = fromBreeze(combinedR.toDenseMatrix)
      val finalQ = if (computeQ) {
        try {
          val invR = inv(combinedR)
          this.multiply(fromBreeze(invR))
        } catch {
          case err: MatrixSingularException =>
            logWarning("R is not invertible and return Q as null")
            null
        }
      } else {
        null
      }
      QRDecomposition(finalQ, finalR)
    }

     def columnSimilaritiesDIMSUM(
                                                 colMags: Array[Double],
                                                 gamma: Double): CoordinateMatrix = {
      require(gamma > 1.0, s"Oversampling should be greater than 1: $gamma")
      require(colMags.size == numCols(), "Number of magnitudes didn't match column dimension")
      val sg = math.sqrt(gamma) // sqrt(gamma) used many times

      // Don't divide by zero for those columns with zero magnitude
      val colMagsCorrected = colMags.map(x => if (x == 0) 1.0 else x)

      val sc = rows.context
      val pBV = sc.broadcast(colMagsCorrected.map(c => sg / c))
      val qBV = sc.broadcast(colMagsCorrected.map(c => math.min(sg, c)))

      val sims = rows.mapPartitionsWithIndex { (indx, iter) =>
        val p = pBV.value
        val q = qBV.value

        val rand = new XORShiftRandom(indx)
        val scaled = new Array[Double](p.size)
        iter.flatMap { row =>
          row match {
            case SparseVector(size, indices, values) =>
              val nnz = indices.size
              var k = 0
              while (k < nnz) {
                scaled(k) = values(k) / q(indices(k))
                k += 1
              }

              Iterator.tabulate(nnz) { k =>
                val buf = new ListBuffer[((Int, Int), Double)]()
                val i = indices(k)
                val iVal = scaled(k)
                if (iVal != 0 && rand.nextDouble() < p(i)) {
                  var l = k + 1
                  while (l < nnz) {
                    val j = indices(l)
                    val jVal = scaled(l)
                    if (jVal != 0 && rand.nextDouble() < p(j)) {
                      buf += (((i, j), iVal * jVal))
                    }
                    l += 1
                  }
                }
                buf
              }.flatten
            case DenseVector(values) =>
              val n = values.size
              var i = 0
              while (i < n) {
                scaled(i) = values(i) / q(i)
                i += 1
              }
              Iterator.tabulate(n) { i =>
                val buf = new ListBuffer[((Int, Int), Double)]()
                val iVal = scaled(i)
                if (iVal != 0 && rand.nextDouble() < p(i)) {
                  var j = i + 1
                  while (j < n) {
                    val jVal = scaled(j)
                    if (jVal != 0 && rand.nextDouble() < p(j)) {
                      buf += (((i, j), iVal * jVal))
                    }
                    j += 1
                  }
                }
                buf
              }.flatten
          }
        }
      }.reduceByKey(_ + _).map { case ((i, j), sim) =>
        MatrixEntry(i.toLong, j.toLong, sim)
      }
      new CoordinateMatrix(sims, numCols(), numCols())
    }



    /** Updates or verifies the number of rows. */
    private def updateNumRows(m: Long) {
      if (nRows <= 0) {
        nRows = m
      } else {
        require(nRows == m,
          s"The number of rows $m is different from what specified or previously computed: ${nRows}.")
      }
    }

}

  object RowMatrixWithGPU {

    /**
      * Fills a full square matrix from its upper triangular part.
      */
    private def triuToFull(n: Int, U: Array[Double]): Matrix = {
      val G = new BDM[Double](n, n)

      var row = 0
      var col = 0
      var idx = 0
      var value = 0.0
      while (col < n) {
        row = 0
        while (row < col) {
          value = U(idx)
          G(row, col) = value
          G(col, row) = value
          idx += 1
          row += 1
        }
        G(col, col) = U(idx)
        idx += 1
        col += 1
      }

      Matrices.dense(n, n, G.data)
    }
  }

