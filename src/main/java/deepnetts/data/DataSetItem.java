package deepnetts.data;

import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.TensorBase;

/**
 * Represents a basic data set item (single row) with input tensor and target
 * vector in a data set.
 */
public class DataSetItem implements MLDataItem {

    private final TensorBase input; // network input
    private final TensorBase targetOutput; // target output for te corresponding input
    private float error; // error measure

   
    public DataSetItem(TensorBase input, TensorBase targetOutput) {
        this.input = input;
        this.targetOutput = targetOutput;
    }

    public DataSetItem(float[] in, float[] targetOut) {
        this.input = new Tensor1D(in);
        this.targetOutput = new Tensor1D(targetOut);
    }

    @Override
    public TensorBase getInput() {
        return input;
    }

    @Override
    public TensorBase getTargetOutput() {
        return targetOutput;
    }

    @Override
    public float getError() {
        return error;
    }

    @Override
    public void setError(float error) {
        this.error = error;
    }

    @Override
    public String toString() {
        return "DataSetItem{" + "input=" + input + ", targetOutput=" + targetOutput + ", error=" + error + '}';
    }

}
