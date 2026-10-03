package deepnetts.net.layers.activation;

import deepnetts.tensor.Tensor;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.Tensor4D;
import java.io.Serializable;

/**
 * Soft sign activation function
 */
public final class SoftSign implements ActivationFunction, Serializable {

    @Override
    public float getValue(final float x) {
       return x / (1 + Math.abs(x));
    }

    /**
     * Note that this method accepts x as param not y as others
     * @param x
     * @return 
     */
    @Override
    public float getPrime(final float x) {
        return 1/(float)Math.pow( (Math.abs(x)+1), 2);
    }
    
    @Override
    public void apply(final Tensor3D tensor, final int channel) { // ovde je problem kada se paralelizuju trebao bih da imam from to idx, ili da ide posle kad zavrse threadovi da cekaju
        final float[] values = tensor.getValues();
        final int chSize = tensor.rows() * tensor.cols();
        final int chStartIdx = chSize * channel;
        final int chEndIdx = chStartIdx + chSize;
        for(int i=chStartIdx; i<chEndIdx; i++) {
            values[i] = values[i] / (1 + Math.abs(values[i]));
        }            
    }
    
    @Override
    public void apply(final Tensor4D tensor, final int batchIdx) {
        final float[] values = tensor.getValues();
        final int batchSize = tensor.depth() *  tensor.rows() * tensor.cols();
        final int batchStartIdx = batchSize * batchIdx;
        final int batchEndIdx = batchStartIdx + batchSize;
        for(int i=batchStartIdx; i<batchEndIdx; i++) {
            values[i] = values[i] / (1 + Math.abs(values[i]));
        }
    }    
    
    @Override
    public void apply(Tensor tensor, int from, int to) { // ovde je problem kada se paralelizuju trebao bih da imam from to idx, ili da ide posle kad zavrse threadovi da cekaju
        final float[] values = tensor.getValues();

        for(int i=from; i<to; i++) {           
            values[i] = values[i] / (1 + Math.abs(values[i]));
       }
    }     
    
}
