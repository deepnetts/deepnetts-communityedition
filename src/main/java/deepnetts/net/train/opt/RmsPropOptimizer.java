package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.util.DeepNettsException;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;
import java.io.Serializable;

/**
 * A variation of AdaDelta optimizer.
 */
public final class RmsPropOptimizer implements Optimizer, Serializable {

    private static final long serialVersionUID = 2804113940324244785L;
    
    private float learningRate; // global learning rate 
    private final TensorBase movAvgGradSqrSum;  
    private Tensor2D movAvgGradSqrSum2D;  
    private Tensor4D movAvgGradSqrSum4D;  
    private final Tensor1D movAvgBiasSqrSum;
    private float beta = 0.9f; // change constant below in formulas

    private float eps = 1e-7f;
    
    public RmsPropOptimizer(AbstractLayer layer) {
        this.learningRate = layer.getLearningRate();
        this.movAvgGradSqrSum = layer.getDeltaWeights().copy();   // mov avg of grads or grad squared
        this.movAvgBiasSqrSum = new Tensor1D(layer.getDeltaBiases().numElements());  
        
        if(movAvgGradSqrSum instanceof Tensor2D) {
            movAvgGradSqrSum2D = (Tensor2D)movAvgGradSqrSum;
        }        
        
        if(movAvgGradSqrSum instanceof Tensor4D) {
            movAvgGradSqrSum4D = (Tensor4D)movAvgGradSqrSum;
        }        
    }    
    

    @Override
    public float calculateDeltaWeight(final float grad, final int... idxs) {
        // TODO: divide by mean sqr http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf slide 29
        if (idxs.length == 2) {
            final float movAvgSum = beta * ((Tensor2D)movAvgGradSqrSum).get(idxs[ROW_IDX], idxs[COL_IDX]) +(1-beta) * grad * grad;
            final float deltaWeight = -(learningRate / ((float)Math.sqrt(movAvgSum+eps))) * grad;            
            ((Tensor2D)movAvgGradSqrSum).set(movAvgSum, idxs[ROW_IDX], idxs[COL_IDX]);
            return deltaWeight;
        }  else if (idxs.length==4) {
            final float movAvgSum = beta * movAvgGradSqrSum4D.get(idxs[0], idxs[1], idxs[2], idxs[3]) + (1-beta) * grad * grad;            
            final float deltaWeight = -(learningRate / ((float) Math.sqrt(movAvgSum+eps))) * grad;
            movAvgGradSqrSum4D.set(movAvgSum,idxs[0], idxs[1], idxs[2], idxs[3]);        
            return deltaWeight;
        } else {
            throw new DeepNettsException("Bad tensor index!");
        }
    }

    @Override
    public float calculateDeltaBias(float grad, int idx) {
        final float movAvgSum = beta * movAvgBiasSqrSum.get(idx) + (1-beta) * grad * grad;
        final float deltaBias = -(learningRate / ((float) Math.sqrt(movAvgSum+eps))) * grad;
        movAvgBiasSqrSum.set(movAvgSum, idx); 
        return deltaBias;
    }
    
    @Override
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }    

    @Override
    public TensorBase calculateDeltaWeight(TensorBase grad) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    @Override
    public Tensor1D calculateDeltaBias(Tensor1D grad) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
    
}