package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.TensorBase;
import java.io.Serializable;

/**
 * Basic Stochastic Gradient Descent optimization algorithm, which iteratively changes weights 
 * in order to find minimum of loss function.
 */
public final class SgdOptimizer implements Optimizer, Serializable  {

    private static final long serialVersionUID = 7408169634780483572L;
        
    private float learningRate;
    private float biasLearningRate;
    
    public SgdOptimizer(AbstractLayer layer) {
        this.learningRate = layer.getLearningRate();
        this.biasLearningRate = this.learningRate;
    }
    
    @Override
    public float calculateDeltaWeight(final float gradient, final int... index) {
        return -learningRate * gradient;
    }

    @Override
    public float calculateDeltaBias(float gradient, int idx) {
        return -biasLearningRate * gradient;
    }
    
    @Override
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }    

    public float getLearningRate() {
        return learningRate;
    }    
    
    public void setBiasLearningRate(float biasLearningRate) {
        this.biasLearningRate = biasLearningRate;
    }
    
    public float getBiasLearningRate() {
        return biasLearningRate;
    }    


    @Override
    public Tensor1D calculateDeltaBias(Tensor1D grad) {
         grad.multiply(-learningRate);
         return grad;
    }

    @Override
    public TensorBase calculateDeltaWeight(TensorBase grad) {
        return grad.multiply(-learningRate);
    }
   
}