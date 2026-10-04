package deepnetts.net.layers.activation;

import deepnetts.core.DeepNetts;
import deepnetts.tensor.Tensor;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.Tensor4D;
import java.io.Serializable;

/**
 * Leaky Rectified Linear Activation and its Derivative.
 *
 * y = x for x > 0, 0.1 * x for x<0
 *        -
 *       | 1, x > 0
 * y' = < | 0.01 , x<=0 - allow a small, positive gradient when the unit is not
 * active https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
 *
 * @author Zoran Sevarac
 */
public final class LeakyRelu implements ActivationFunction, Serializable {

        
    private final float a;

    public LeakyRelu() {
        this.a = 0.01f;
    }

    public LeakyRelu(float a) {
        this.a = a;
    }

    @Override
    public float getValue(final float x) {
        return (x >= 0 ? x : 0.01f * x);
    }

    @Override
    public float getPrime(final float y) {
        return (y > 0 ? 1 : a);
    }

    @Override
    public void apply(final Tensor3D tensor, final int channel) { // ovde je problem kada se paralelizuju trebao bih da imam from to idx, ili da ide posle kad zavrse threadovi da cekaju
        final float[] values = tensor.getValues();
        final int chSize = tensor.rows() * tensor.cols();
        final int chStartIdx = chSize * channel;
        final int chEndIdx = chStartIdx + chSize;
        for (int i = chStartIdx; i < chEndIdx; i++) {
            values[i] = (values[i] >= 0 ? values[i] : 0.01f * values[i]);
        }
    }

    @Override
    public void apply(final Tensor4D tensor, final int batchIdx) {
        final float[] values = tensor.getValues();
        final int batchSize = tensor.depth() * tensor.rows() * tensor.cols();
        final int batchStartIdx = batchSize * batchIdx;
        final int batchEndIdx = batchStartIdx + batchSize;
        for (int i = batchStartIdx; i < batchEndIdx; i++) {
            values[i] = (values[i] >= 0 ? values[i] : 0.01f * values[i]);
        }
    }

    @Override
    public void apply(Tensor tensor, int from, int to) { // ovde je problem kada se paralelizuju trebao bih da imam from to idx, ili da ide posle kad zavrse threadovi da cekaju

        if (DeepNetts.getInstance().useVectorAPI()) {
       //     vectorizationImpl.leakyRelu(tensor, a);
            return;
        }
        final float[] values = tensor.getValues();

        for (int i = from; i < to; i++) {
            values[i] = (values[i] >= 0 ? values[i] : 0.01f * values[i]);
        }
    }

//    private void applyVectorized(Tensor tensor) {
//
//        int upperBound = SPECIES.loopBound(tensor.numElements());
//
//        FloatVector alphaVector = FloatVector.broadcast(SPECIES, a);
//
//        final float[] values = tensor.getValues();
//
//        for (int offset = 0; offset < upperBound; offset += vecLen) {
//            FloatVector v = FloatVector.fromArray(SPECIES, values, offset);
//            VectorMask<Float> mask = v.compare(VectorOperators.LT, 0.0f);
//            FloatVector scaled = v.mul(alphaVector);
//            FloatVector result = v.blend(scaled, mask);
//            result.intoArray(values, offset);
//        }
//
//        for (int i = upperBound; i < tensor.numElements(); i++) {
//            values[i] = values[i] > 0 ? values[i] : a * values[i];
//        }
//    }

}
