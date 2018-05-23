/**
  * Created by Taehee on 2018. 5. 22..
  */



import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import akka.stream.scaladsl.{Flow, Framing, Sink, Source, Tcp}
import akka.stream.scaladsl.Tcp.{IncomingConnection, ServerBinding}
import akka.util.ByteString

import scala.concurrent.Future

object InputStream {

  implicit val system = ActorSystem("QuickStart")
  implicit val materializer = ActorMaterializer()

  def libSvmParser(libSvm:String) = {
    val split = libSvm.split(" +")
    val label = split.take(1)(0) toInt
    val feature = split.drop(1).map{s=>
      val kv = s.split(":")
      val (k, v) = (kv(0), kv(1))
      (k.toInt -> v.toDouble)
    }.toMap
    (label, FtrlRun.mapToSparseVector(feature, Int.MaxValue))
  }

  val ftrl = new Ftrl().setAlpha(5).setBeta(1).setL1(1.5).setL2(0)









  def main(args: Array[String]): Unit = {


    //val aa = echo -n "1 0:2 3:4 5:6" | nc 127.0.0.1 8888
    //val aa = msg   /SERVER:127.0.0.1 8888 "1 0:2 3:4 5:6"
    val connections: Source[IncomingConnection, Future[ServerBinding]] = Tcp().bind("localhost", 8888)

    connections runForeach { connection ⇒
      println(s"New connection from: ${connection.remoteAddress}")

      val echo = Flow[ByteString].via(Framing.delimiter(
        ByteString("\n"),
        maximumFrameLength = 2048,
        allowTruncation = true))
        .map{x=>
          libSvmParser(x.utf8String)
        }
        .map{x=>
          ftrl.update(x)
          ftrl.weight.toString() + "\n"
        }
        .map(ByteString(_))

      connection.handleWith(echo)
    }





  }
}
