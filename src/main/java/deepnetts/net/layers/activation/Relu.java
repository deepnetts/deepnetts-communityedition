package deepnetts.net.layers.activation;

import deepnetts.core.DeepNetts;
import deepnetts.tensor.Tensor;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.Tensor4D;
import java.io.Serializable;


/**
 * Rectified Linear Activation and its Derivative.
 *
 * y = max(0, x) - | 1, x > 0 y' = < | 0, 0x<=0 -
 *
 * @author Zoran Sevarac
 */
public final class Relu implements ActivationFunction, Serializable {

    private float clipValue = 3f;// should be clipped in all methods below
    
    @Override
    public float getValue(final float x) {
        //Math.min(Math.max(0, x), clipValue);
        return Math.max(0, x);
    }

    @Override
    public float getPrime(final float y) {
        return (y > 0 ? 1 : 0);
    }

    @Override
    public void apply(final Tensor3D tensor, final int channel) { // ovde je problem kada se paralelizuju trebao bih da imam from to idx, ili da ide posle kad zavrse threadovi da cekaju
        final float[] values = tensor.getValues();
        final int chSize = tensor.rows() * tensor.cols(); // ovo je za 3D vektore
        final int chStartIdx = chSize * channel;
        final int chEndIdx = chStartIdx + chSize;
        for (int i = chStartIdx; i < chEndIdx; i++) {
            values[i] = Math.max(0, values[i]);
        }
    }

    @Override
    public void apply(final Tensor4D tensor, final int batchIdx) {
        final float[] values = tensor.getValues();
        final int batchSize = tensor.depth() * tensor.rows() * tensor.cols();
        final int batchStartIdx = batchSize * batchIdx;
        final int batchEndIdx = batchStartIdx + batchSize;
        for (int i = batchStartIdx; i < batchEndIdx; i++) {
            values[i] = Math.max(0, values[i]);
        }
    }

    @Override
    public void apply(Tensor tensor, int from, int to) {

        if (DeepNetts.getInstance().useVectorAPI()) {
//            vectorizationImpl.relu(tensor);
            return;
        }

        final float[] values = tensor.getValues();

        for (int i = from; i < to; i++) {
            values[i] = Math.max(0, values[i]);
        }
    }

    // @TODOL add   apply(TensorBase tensor) for all elements in tensor
//    private void applyVectorized(Tensor tensor) {
//
//        final int upperBound = SPECIES.loopBound(tensor.numElements());
//
//        final float[] values = tensor.getValues();
//
//        for (int offset = 0; offset < upperBound; offset += vecLen) {
//            FloatVector v = FloatVector.fromArray(SPECIES, values, offset);
//            v.max(0.0f).intoArray(values, offset);
//        }
//
//        for (int i = upperBound; i < tensor.numElements(); i++) {
//            values[i] = Math.max(0.0f, values[i]);
//        }
//    }
}
