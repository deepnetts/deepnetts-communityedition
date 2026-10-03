/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package deepnetts.tensor;

   // ovo prebaci u tensor base najverovatnije - na osnovu ovoga zna sta koja pozicija u shapu predstavlja, ili mozda nije potrebno nego je to samo fizicki layout - razmis
import java.io.Serializable;

    // ovo zapravo nije fizicki nego logicki layut u smislu redolseda dimenzija/indeksa u shape-u
  // na osnovu ovoga zna kako da izracuna idx - proveri kako radu cudnn, tf i pt i uradi najslicnije sto moze
    // gde mi ovo treba? u shapeu?
   public class Layout implements Serializable {
       public static final Layout NCHW = new Layout(0, 1, 2, 3);
       public static final Layout NHWC = new Layout(0, 3, 1, 2);
       
       public final int nIdx, cIdx,  hIdx, wIdx;   
       // batchIdx, rowIdx, colIdx, chIdx
        
        public Layout(int nIdx, int cIdx, int hIdx, int wIdx) {
            this.nIdx = nIdx;
            this.cIdx = cIdx;
            this.hIdx = hIdx;
            this.wIdx = wIdx;
        }

   }
