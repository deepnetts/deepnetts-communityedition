/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package deepnetts.examples.util;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamClass;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.UUID;

/**
 *
 * @author Zoran
 */
public class PrintUIDs extends ObjectInputStream {

  public PrintUIDs(InputStream in) throws IOException {
    super(in);
  }

  @Override
  protected ObjectStreamClass readClassDescriptor() throws IOException,
      ClassNotFoundException {
    ObjectStreamClass descriptor = super.readClassDescriptor();
    System.out.println("name=" + descriptor.getName());
    System.out.println("serialVersionUID=" + descriptor.getSerialVersionUID());
    return descriptor;
  }

  public static void main(String[] args) throws IOException,
      ClassNotFoundException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    ObjectOutputStream oos = new ObjectOutputStream(baos);
    List<Object> list = Arrays.asList((Object) new Date(), UUID.randomUUID());
    oos.writeObject(list);
    oos.close();
    InputStream in = new ByteArrayInputStream(baos.toByteArray());
    ObjectInputStream ois = new PrintUIDs(in);
    ois.readObject();
  }

}	