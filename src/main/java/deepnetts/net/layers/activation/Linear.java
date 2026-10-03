package deepnetts.net.layers.activation;

import deepnetts.tensor.Tensor;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.Tensor4D;
import java.io.Serializable;

/**
 * Linear activation function and its derivative.
 * Commonly used as an activation function in output layer for regression(numeric prediction) tasks.
 * By default it passes it's input to output y = k * x where k is given as slope parameter.
 * 
 * y = slope * x
 * y' = slope
 * 
 * slope parameter = 1 by default.
 * 
 * @see ActivationFunction
 */
public final class Linear implements ActivationFunction, Serializable {
    
    final int slope;

    public Linear() {
        this.slope = 1;
    }    
    
    public Linear(int slope) {
        this.slope = slope;
    }
    
    @Override
    public float getValue(final float x) {
        return slope * x;
    }

    @Override
    public float getPrime(final float y) {
        return slope;
    }    
    
    @Override
    public void apply(final Tensor3D tensor, final int channel) { // ovde je problem kada se paralelizuju trebao bih da imam from to idx, ili da ide posle kad zavrse threadovi da cekaju
        final float[] values = tensor.getValues();
        final int chSize = tensor.rows() * tensor.cols();
        final int chStartIdx = chSize * channel;
        final int chEndIdx = chStartIdx + chSize;
        for(int i=chStartIdx; i<chEndIdx; i++) {
            values[i] = slope * values[i];
        }
    }
    
    @Override
    public void apply(final Tensor4D tensor, final int batchIdx) {
        final float[] values = tensor.getValues();
        final int batchSize = tensor.depth() *  tensor.rows() * tensor.cols();
        final int batchStartIdx = batchSize * batchIdx;
        final int batchEndIdx = batchStartIdx + batchSize;
        for(int i=batchStartIdx; i<batchEndIdx; i++) {
            values[i] = slope * values[i];
        }
    }    
    
    
    @Override
    public void apply(Tensor tensor, int from, int to) { // ovde je problem kada se paralelizuju trebao bih da imam from to idx, ili da ide posle kad zavrse threadovi da cekaju
        final float[] values = tensor.getValues();

        for(int i=from; i<to; i++) {
            values[i] = slope * values[i];
        }
    }       
}