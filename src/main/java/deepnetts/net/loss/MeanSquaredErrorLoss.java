package deepnetts.net.loss;

import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.Layer;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.TensorBase;
import java.io.Serializable;

/**
 * Mean Squared Error Loss function. Sum squared errors over all input patterns and
 * all outputs. Should be used for regression problems.
 *
 * Math formula:
 *                       N   K 
 *      E = 1/(2*N*K) * SUM(SUM(y-t)^2) + regSum
 *
 * where N is number of patterns and K is dimension of output vector, and regSum is L1 or L2 regularization multiplied with lambda.
 *  
 * Bishop, pg. 89, eq. 3.34
 * Also recommended this formula in Proben1 Technical report
 * 
 * @see LossFunction
 * @see CrossEntropyLoss
 */
public final class MeanSquaredErrorLoss implements LossFunction, Serializable {
    
    private static final long serialVersionUID  = 6004545721099801809L;
    
    private TensorBase outputErrorTsr; // moze biti 1D ili 2D
    private float totalError = 0;
    private int patternCount = 0;
    private float regularizationSum = 0;
    private float patternLoss=0;
    private TensorBase patternLossTsr; 
    private int batchSize = 1;
    private Tensor2D predictedOut2D;
    
    /**
     * Creates a new mean squared error loss for the given neural network.
     * 
     * @param neuralNet 
     */
    public MeanSquaredErrorLoss(NeuralNetwork neuralNet) {
       // outputError = new float[neuralNet.getOutputLayer().getWidth()];
       boolean isBatch = neuralNet.getInputLayer().getBatchSize()> 1;
        TensorBase networkOutput = neuralNet.getOutputLayer().getOutputs(); // zasto je ovaj null jer jo snije zavrsio kreiranje mreze
        if  (!isBatch) { //(networkOutput instanceof Tensor1D) {
            int width = neuralNet.getOutputLayer().getWidth();
            outputErrorTsr = new Tensor1D(width);
        } else { // if (networkOutput instanceof Tensor2D) {            
            int outWidth = neuralNet.getOutputLayer().getWidth();
            batchSize = neuralNet.getInputLayer().getBatchSize();
            patternLossTsr = new Tensor2D(outWidth, batchSize);
            outputErrorTsr = new Tensor2D(outWidth, batchSize);
        }        
    }
     

    /**
     * Adds output error vector for the given predicted and target output vectors
     * to total error sum and returns and  error vector.
     *
     * @param predictedOutput   
     * @param targetOutput  
     * @return
     */
    /*
    @Override
    public float[] addPatternError(final float[] predictedOutput, final float[] targetOutput) {
        for (int i = 0; i < predictedOutput.length; i++) {
            outputError[i] = predictedOutput[i] - targetOutput[i];
            patternLoss = outputError[i] * outputError[i];
            totalError += patternLoss;
        }

        patternCount++;
        return outputError;
    }
    */
    
    @Override
    public TensorBase addPatternError(TensorBase predictedOut, TensorBase targetOut) {
        // za 1d i 
        if (predictedOut instanceof Tensor1D) {
            predictedOut.sub(targetOut, outputErrorTsr);
            patternLoss = outputErrorTsr.sumSqr();
            totalError += patternLoss; 
            // same as code below
//            for (int i = 0; i < predictedOutput.length; i++) {
//                outputError[i] = predictedOutput[i] - targetOutput[i];
//                patternLoss = outputError[i] * outputError[i];
//                totalError += patternLoss;
//            }

            patternCount++;
        } else if (predictedOut instanceof Tensor2D) { // 2d/batch out
            predictedOut2D = (Tensor2D)predictedOut;
            predictedOut.sub(targetOut, outputErrorTsr);
            totalError += outputErrorTsr.sumSqr();
            patternCount += predictedOut2D.cols();
        }
        
        
        return outputErrorTsr;        
    }    

    /**
     * Add regularization sum to total loss
     * 
     * @param regSum regularization sum
     */
    @Override
    public void addRegularizationSum(final float regSum) {
       regularizationSum = regSum;
    }    
    
    @Override
    public float getTotal() {
        // a sta ako je mini batch ond aje patternCount drugaciji - analiziraj loss u mini batch modu i online modu
        if (outputErrorTsr instanceof Tensor1D) {
            return ( totalError / (2 * patternCount * outputErrorTsr.numElements() ) ) + regularizationSum; // iz ng slajdova i regularizacija bi trebalo da se deli sa 1/2*patternCount
        } else if (outputErrorTsr instanceof Tensor2D) {
            return ( totalError / (2 * patternCount * ((Tensor2D)outputErrorTsr).rows() ) ) + regularizationSum;
        } else {
            throw new IllegalStateException("Illegal tensor dimensions");
        }
        // https://www.coursera.org/learn/deep-neural-network/lecture/qcogH/mini-batch-gradient-descent 9:08 DELI REGULARIZACIJU SA 1/2N
    }

    @Override
    public void reset() {
        totalError = 0;
        patternCount = 0;
        regularizationSum = 0;
    }

    @Override
    public float getPatternLoss() {
        return patternLoss;
    }



}
