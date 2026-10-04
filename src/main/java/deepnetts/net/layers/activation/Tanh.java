package deepnetts.net.layers.activation;

import deepnetts.core.DeepNetts;
import deepnetts.tensor.Tensor;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.Tensor4D;
import java.io.Serializable;

/**
 * Hyperbolic tangens activation function
 * 
 * @author zoran
 */
public final class Tanh implements ActivationFunction, Serializable {
        
    @Override
    public float getValue(float x) {
       if (x > 8.5f) x=8.5f;
       if (x < -8.5f) x=-8.5f;
       
       final float e2x = (float)Math.exp(2*x);   
       return (e2x-1) / (e2x+1); // calculate tanh 
    }

    @Override
    public float getPrime(final float y) {
        return (1-y*y);
    }
    
    @Override
    public void apply(final Tensor3D tensor, final int channel) { // ovde je problem kada se paralelizuju trebao bih da imam from to idx, ili da ide posle kad zavrse threadovi da cekaju
        final float[] values = tensor.getValues();
        final int chSize = tensor.rows() * tensor.cols();
        final int chStartIdx = chSize * channel;
        final int chEndIdx = chStartIdx + chSize;
        for(int i=chStartIdx; i<chEndIdx; i++) {
            if (values[i] > 8.5f) values[i]=8.5f;
            if (values[i] < -8.5f) values[i]=-8.5f;
            final float e2x = (float)Math.exp(2*values[i]);    
            values[i] = (e2x-1) / (e2x+1); // calculate tanh
        }            
    }
    
    @Override
    public void apply(final Tensor4D tensor, final int batchIdx) {
        final float[] values = tensor.getValues();
        final int batchSize = tensor.depth() *  tensor.rows() * tensor.cols();
        final int batchStartIdx = batchSize * batchIdx;
        final int batchEndIdx = batchStartIdx + batchSize;
        for(int i=batchStartIdx; i<batchEndIdx; i++) {
            if (values[i] > 8.5f) values[i]=8.5f;
            if (values[i] < -8.5f) values[i]=-8.5f;
            final float e2x = (float)Math.exp(2*values[i]);    
            values[i] = (e2x-1) / (e2x+1); // calculate tanh
        }
    }       
    
    @Override
    public void apply(Tensor tensor, int from, int to) { // ovde je problem kada se paralelizuju trebao bih da imam from to idx, ili da ide posle kad zavrse threadovi da cekaju
        
        if (DeepNetts.getInstance().useVectorAPI()) {           
   //         vectorizationImpl.tanh(tensor);
            return;
        }
        
        final float[] values = tensor.getValues();

        for(int i=from; i<to; i++) {
            if (values[i] > 8.5f) values[i]=8.5f;
            if (values[i] < -8.5f) values[i]=-8.5f;            
            final float e2x = (float)Math.exp(2*values[i]);    
            values[i] = (e2x-1) / (e2x+1); // calculate tanh
       }
    }     
    
    
//    private void applyVectorized(Tensor tensor){
//    
//        int upperBound = SPECIES.loopBound(tensor.numElements());
//        
//        final float[] values = tensor.getValues();
//        
//        for (int offset = 0; offset < upperBound; offset+= vecLen) {
//            FloatVector v = FloatVector.fromArray(SPECIES, values, offset);
//            v.lanewise(VectorOperators.TANH).intoArray(values, offset);
//        }
//        
//        for (int i = upperBound; i < tensor.numElements(); i++) {
//            values[i] = (float) Math.tanh(values[i]);
//        }
//    
//    }
}
