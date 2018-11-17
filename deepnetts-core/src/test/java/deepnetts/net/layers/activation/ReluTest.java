package deepnetts.net.layers.activation;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Relu Activation Functio Test.
 * 
 * @author Zoran Sevarac
 */
public class ReluTest {
    
   Relu instance; 
    
   @Before
   public void setUp() {
      instance = new Relu();
   }

    /**
     * Test of getValue method, of class Relu.
     */
    @Test
    public void testGetValue() {
        float[] x = {-7.0f,  -6.9f,  -6.8f,  -6.7f,  -6.6f,  -6.5f,  -6.4f,  -6.3f,  -6.2f,  -6.1f,  -6.0f,  -5.9f,  -5.8f,  -5.7f,  -5.6f,  -5.5f,  -5.4f,  -5.3f,  -5.2f,  -5.1f,  -5.0f,  -4.9f,  -4.8f,  -4.7f,  -4.6f,  -4.5f,  -4.4f,  -4.3f,  -4.2f,  -4.1f,  -4.0f,  -3.9f,  -3.8f,  -3.7f,  -3.6f,  -3.5f,  -3.4f,  -3.3f,  -3.2f,  -3.1f,  -3.0f,  -2.9f,  -2.8f,  -2.7f,  -2.6f,  -2.5f,  -2.4f,  -2.3f,  -2.2f,  -2.1f,  -2.0f,  -1.9f,  -1.8f,  -1.7f,  -1.6f,  -1.5f,  -1.4f,  -1.3f,  -1.2f,  -1.1f,  -1.0f,  -0.9f,  -0.8f,  -0.7f,  -0.6f,  -0.5f,  -0.4f,  -0.3f,  -0.2f,  -0.1f,  0f,  0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  0.6f,  0.7f,  0.8f,  0.9f,  1.0f,  1.1f,  1.2f,  1.3f,  1.4f,  1.5f,  1.6f,  1.7f,  1.8f,  1.9f,  2.0f,  2.1f,  2.2f,  2.3f,  2.4f,  2.5f,  2.6f,  2.7f,  2.8f,  2.9f,  3.0f,  3.1f,  3.2f,  3.3f,  3.4f,  3.5f,  3.6f,  3.7f,  3.8f,  3.9f,  4.0f,  4.1f,  4.2f,  4.3f,  4.4f,  4.5f,  4.6f,  4.7f,  4.8f,  4.9f,  5.0f,  5.1f,  5.2f,  5.3f,  5.4f,  5.5f,  5.6f,  5.7f,  5.8f,  5.9f,  6.0f,  6.1f,  6.2f,  6.3f,  6.4f,  6.5f,  6.6f,  6.7f,  6.8f,  6.9f,  7.0f};
        float[] y = {0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f, 0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f,  0f, 0f,  0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  0.6f,  0.7f,  0.8f,  0.9f,  1.0f,  1.1f,  1.2f,  1.3f,  1.4f,  1.5f,  1.6f, 1.7f,  1.8f,  1.9f,  2.0f,  2.1f,  2.2f,  2.3f,  2.4f,  2.5f,  2.6f,  2.7f,  2.8f,  2.9f,  3.0f,  3.1f,  3.2f,  3.3f,  3.4f,  3.5f,  3.6f,  3.7f,  3.8f,  3.9f,  4.0f,  4.1f,  4.2f,  4.3f,  4.4f,  4.5f,  4.6f,  4.7f,  4.8f, 4.9f,  5.0f,  5.1f,  5.2f,  5.3f,  5.4f,  5.5f,  5.6f,  5.7f,  5.8f,  5.9f,  6.0f, 6.1f,  6.2f,  6.3f,  6.4f,  6.5f,  6.6f,  6.7f, 6.8f, 6.9f, 7.0f};
               
        for (int i = 0; i < x.length; i++) {
            float expResult = y[i];
            float result = instance.getValue(x[i]);
            assertEquals(expResult, result, 0);
        } 
    }

    /**
     * Test of getPrime method, of class Relu.
     */
    @Test
    public void testGetPrime() {
        float[] x = {-7.0f,  -6.9f,  -6.8f,  -6.7f,  -6.6f,  -6.5f,  -6.4f,  -6.3f,  -6.2f,  -6.1f,  -6.0f,  -5.9f,  -5.8f,  -5.7f,  -5.6f,  -5.5f,  -5.4f,  -5.3f,  -5.2f,  -5.1f,  -5.0f,  -4.9f,  -4.8f,  -4.7f,  -4.6f,  -4.5f,  -4.4f,  -4.3f,  -4.2f,  -4.1f,  -4.0f,  -3.9f,  -3.8f,  -3.7f,  -3.6f,  -3.5f,  -3.4f,  -3.3f,  -3.2f,  -3.1f,  -3.0f,  -2.9f,  -2.8f,  -2.7f,  -2.6f,  -2.5f,  -2.4f,  -2.3f,  -2.2f,  -2.1f,  -2.0f,  -1.9f,  -1.8f,  -1.7f,  -1.6f,  -1.5f,  -1.4f,  -1.3f,  -1.2f,  -1.1f,  -1.0f,  -0.9f,  -0.8f,  -0.7f,  -0.6f,  -0.5f,  -0.4f,  -0.3f,  -0.2f,  -0.1f,  0f,  0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  0.6f,  0.7f,  0.8f,  0.9f,  1.0f,  1.1f,  1.2f,  1.3f,  1.4f,  1.5f,  1.6f,  1.7f,  1.8f,  1.9f,  2.0f,  2.1f,  2.2f,  2.3f,  2.4f,  2.5f,  2.6f,  2.7f,  2.8f,  2.9f,  3.0f,  3.1f,  3.2f,  3.3f,  3.4f,  3.5f,  3.6f,  3.7f,  3.8f,  3.9f,  4.0f,  4.1f,  4.2f,  4.3f,  4.4f,  4.5f,  4.6f,  4.7f,  4.8f,  4.9f,  5.0f,  5.1f,  5.2f,  5.3f,  5.4f,  5.5f,  5.6f,  5.7f,  5.8f,  5.9f,  6.0f,  6.1f,  6.2f,  6.3f,  6.4f,  6.5f,  6.6f,  6.7f,  6.8f,  6.9f,  7.0f};
        float[] y = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 };
        
        for (int i = 0; i < x.length; i++) {
            float expResult = y[i];
            float result = instance.getPrime(x[i]);
            assertEquals(expResult, result, 0);
        } 
    }
    
    
}
