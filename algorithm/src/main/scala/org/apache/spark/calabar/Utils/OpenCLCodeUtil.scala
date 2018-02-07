package org.apache.spark.calabar.Utils

import org.apache.spark.internal.Logging

import scala.io.Source


/**
  * Created by wx on 2017/4/18.
  */
object OpenCLCodeUtil {

  val fileSeparator: String = System.getProperty("file.separator")
  val path: String =Thread.currentThread().getContextClassLoader.getResource("").getPath+"algorithm"
  def get(algorithmName:String):String={

    val file=Source.fromFile(path+fileSeparator+algorithmName+".cl","utf-8")
    file.mkString
  }

}
