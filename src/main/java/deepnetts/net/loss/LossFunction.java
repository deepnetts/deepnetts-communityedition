package deepnetts.net.loss;


import deepnetts.net.NeuralNetwork;
import javax.visrec.ml.data.DataSet;
import deepnetts.data.MLDataItem;
import deepnetts.tensor.Tensor;
import deepnetts.tensor.TensorBase;

/**
 * Base Interface for all loss functions.
 * Loss function is a component of a deep learning algorithm which calculates an error,
 * as a difference of actual (or predicted) and desired (target) output of a neural network.
 * The total error for some training is usually calculated as a average of errors for all individual input-output pairs.
 * The higher value of loss function, means higher error and lower accuracy of prediction.
 *
 * @see MeanSquaredErrorLoss
 * @see BinaryCrossEntropyLoss
 * @see CrossEntropyLoss
 */
public interface LossFunction {

    /**
     * Calculates pattern error for singe pattern for the specified predicted and target outputs,
     * adds the error to total error, and returns the pattern error.
     *
     * @param predictedOutput predicted/actual network output vector
     * @param targetOutput target network output vector
     * @return error vector error vector for the given predicted and target vectors
     */
    //public float[] addPatternError(float[] predictedOutput, float[] targetOutput);
    
    public TensorBase addPatternError(TensorBase predictedOutput, TensorBase targetOutput);

    /**
     * Adds specified regularization sum to total loss.
     * @param regSum regularization sum
     */
    public void addRegularizationSum(float regSum);

    /**
     * Returns the total error calculated by this loss function.
     *
     * @return total error calculated by this loss function
     */
    public float getTotal();
    
    public float getPatternLoss();

    /**
     * Resets the total error and pattern counter.
     */
    public void reset();

    /**
     * Calculates and returns loss function value for the given neural network and data set.
     *
     * @param nnet
     * @param dataSet
     * @return
     */
    default public float valueFor(NeuralNetwork nnet, DataSet<? extends MLDataItem> dataSet) {
        reset();

        for(MLDataItem dsItem : dataSet) {
            nnet.setInput(dsItem.getInput());
            TensorBase output = nnet.getOutputAsTensor();
            addPatternError(output, dsItem.getTargetOutput());
        }
        return getTotal();
    }

}
