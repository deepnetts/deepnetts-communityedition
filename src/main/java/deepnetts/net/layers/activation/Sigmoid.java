package deepnetts.net.layers.activation;

import deepnetts.core.DeepNetts;
import deepnetts.tensor.Tensor;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.Tensor4D;
import java.io.Serializable;

/**
 * Sigmoid activation function
 *
 * @see
 * <a href=" https://en.wikipedia.org/wiki/Sigmoid_function">Sigmoid_function on
 * Wikipedia</a>
 * @author Zoran Sevarac
 */
public final class Sigmoid implements ActivationFunction, Serializable {
// 	https://shaktiwadekar.medium.com/how-to-avoid-numerical-overflow-in-sigmoid-function-numerically-stable-sigmoid-function-5298b14720f6
// https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth


    
    /*
        Numerical stability 
        if x == 89 y = infinity 
        x mora biti manje od 89 da bi bla numericki stabilna za X > 0
        if x >104 onda e^-x=0 
        if x < -88 then e^-x = infinity // ova mi je problematicna
     */
    @Override
    public float getValue(final float x) {
        //if (x < -88) return 0.9999f;
        //else if (x > 104) return 0.9999f;

        return 1 / (1 + (float) Math.exp(-x));
    }

    @Override
    public float getPrime(final float y) {
        return y * (1 - y);
    }

    @Override
    public void apply(final Tensor3D tensor, final int channel) { // ovde je problem kada se paralelizuju trebao bih da imam from to idx, ili da ide posle kad zavrse threadovi da cekaju
        final float[] values = tensor.getValues();
        final int chSize = tensor.rows() * tensor.cols();
        final int chStartIdx = chSize * channel;
        final int chEndIdx = chStartIdx + chSize;
        for (int i = chStartIdx; i < chEndIdx; i++) {
            values[i] = 1 / (1 + (float) Math.exp(-values[i]));
        }
    }

    @Override
    public void apply(final Tensor4D tensor, final int batchIdx) {
        final float[] values = tensor.getValues();
        final int batchSize = tensor.depth() * tensor.rows() * tensor.cols();
        final int batchStartIdx = batchSize * batchIdx;
        final int batchEndIdx = batchStartIdx + batchSize;
        for (int i = batchStartIdx; i < batchEndIdx; i++) {
            values[i] = 1 / (1 + (float) Math.exp(-values[i]));
        }
    }

    @Override
    public void apply(Tensor tensor, int from, int to) { // ovde je problem kada se paralelizuju trebao bih da imam from to idx, ili da ide posle kad zavrse threadovi da cekaju

        if (DeepNetts.getInstance().useVectorAPI()) {
        //   vectorizationImpl.sigmoid(tensor);
            return;
        }

        final float[] values = tensor.getValues();

        for (int i = from; i < to; i++) {
            values[i] = 1 / (1 + (float) Math.exp(-values[i]));
        }
    }

//    private void applyVectorized(Tensor tensor) {
//
//        int upperBound = SPECIES.loopBound(tensor.numElements());
//
//        final float[] values = tensor.getValues();
//        FloatVector onesVector = FloatVector.broadcast(SPECIES, 1.0f);
//
//        // s(x) = 1 / ( 1 + e^(-x))
//        for (int offset = 0; offset < upperBound; offset += vecLen) {
//            FloatVector v = FloatVector.fromArray(SPECIES, values, offset);
//            // neg = -x
//            FloatVector vNegative = v.neg();
//            // exp(x) = e^x
//            FloatVector exp = vNegative.lanewise(VectorOperators.EXP);
//            FloatVector sigmoid = onesVector.div(exp.add(1.0f));
//            sigmoid.intoArray(values, offset);
//        }
//
//        for (int i = upperBound; i < tensor.numElements(); i++) {
//            values[i] = 1 / (1 + (float) Math.exp(-values[i]));
//        }
//    }

}
