/**
  * Created by Taehee on 2018. 5. 31..
  */

object AvazuStream {



  import java.io.{FileInputStream, FileOutputStream, PrintStream}
  import java.util.zip.GZIPInputStream

  import scala.io.Source



  def main(args: Array[String]): Unit = {


    val reader = new GZIPInputStream(new FileInputStream("~/stream/avazu/train.gz"))
    val writer = new PrintStream(new FileOutputStream("~/stream/avazu/avazu-train", true))
    //val filename = "/Users/Taehee/Downloads/test"
    for (line <- Source.fromInputStream(reader).getLines) {

      println(line)
      writer.append(line + "\n")
      Thread.sleep(10)

    }
    writer.close()

  }
}
