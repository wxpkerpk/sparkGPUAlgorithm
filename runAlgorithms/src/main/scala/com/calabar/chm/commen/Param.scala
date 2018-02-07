package com.calabar.chm.commen

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.Saveable
import org.apache.spark.sql.Dataset

import scala.collection.mutable.ArrayBuffer

/**
  * Created by wx on 2017/6/1.
  */
class Param(value: Any, name: Any, desc: Any) {

  def ::(x: Param) = {

    val list = new ArrayBuffer[Param]()
    list += this
    list += x
    list
  }

}

object Param {
  def apply(value: Any, name: Any, desc: Any): Param = new Param(value, name, desc)

}





