   package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.tensor.Tensor;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;
import java.io.Serializable;


/**
 * Momentum optimization adds momentum parameter to basic Stochastic Gradient Descent, which can accelerate the process. 
 */
public final class MomentumOptimizer implements Optimizer, Serializable {
   
    private static final long serialVersionUID = 6936741415174730939L;
    
    private final float momentum;
    private float learningRate;    

    private final TensorBase prevDeltaWeights;
    private Tensor2D prevDeltaWeights2D;    
    private Tensor4D prevDeltaWeights4D;    
    private final Tensor1D prevDeltaBiases;

    // https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/
    // https://www.scaler.com/topics/momentum-based-gradient-descent/

   
    public MomentumOptimizer(AbstractLayer layer) { // <O, W>
        this.learningRate = layer.getLearningRate();
        this.momentum = layer.getMomentum();
        this.prevDeltaWeights = layer.getPrevDeltaWeights();
        
        if(prevDeltaWeights instanceof Tensor2D) {
            prevDeltaWeights2D = (Tensor2D)prevDeltaWeights;
        }        
        
        if(prevDeltaWeights instanceof Tensor4D) {
            prevDeltaWeights4D = (Tensor4D)prevDeltaWeights;
        }
        this.prevDeltaBiases = layer.getPrevDeltaBiases();
    }
    
    @Override
    public float calculateDeltaWeight(final float grad, final int... idxs) {
        if (idxs.length==2)
            return -learningRate * grad + momentum * prevDeltaWeights2D.get(idxs[ROW_IDX], idxs[COL_IDX]); 
        else if (idxs.length==4) {
            final float dWeight = -learningRate * grad + momentum * prevDeltaWeights4D.get(idxs[0], idxs[1], idxs[2], idxs[3]);
            return dWeight;
        }
        else
            throw new IllegalStateException("Delta weight tensor dimensins error!");
    }
    
    @Override
    public TensorBase calculateDeltaWeight(TensorBase grad) {
        // grad su kopije 
        return grad.multiply(-learningRate).add(prevDeltaWeights.multiply(momentum));
        
    }    

    @Override
    public float calculateDeltaBias(float grad, int idx) {
         return -learningRate * grad + momentum * prevDeltaBiases.get(idx);
    }
    
    @Override
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }    

    @Override
    public Tensor1D calculateDeltaBias(Tensor1D grad) {
        return (Tensor1D)grad.multiply(-learningRate).add((Tensor1D)prevDeltaBiases.multiply(momentum));
    }

}