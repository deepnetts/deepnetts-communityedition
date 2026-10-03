package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.TensorBase;
import deepnetts.util.DeepNettsException;

/**
 * Optimization technique to tune network's weights parameters used by training algorithm.
 */
public interface Optimizer {
    /**
    * Smoothing term to prevent division by zero if sqr grad sum becomes zero   1e-8 should be also tried
    * https://d2l.ai/chapter_optimization/adagrad.html   1e-6
    * The value to use is 1e-6, 1e-8, Keras uses 1e-7 for adam
    */
//    public float EPS = 1e-6f;
    
    public float calculateDeltaWeight(final float gradient, final int... index);
   
    public TensorBase calculateDeltaWeight(final TensorBase grad);

    
    public float calculateDeltaBias(final float gradient, final int idx);
    
    public Tensor1D calculateDeltaBias(final Tensor1D grad);
    
    public void setLearningRate(float learningRate);
    
    /**
     * Factory method to create different types of optimizers
     * @param type
     * @param layer
     * @return 
     */
    public static Optimizer create(OptimizerType type, AbstractLayer layer) {
        switch (type) {
            case SGD:
                return new SgdOptimizer(layer);
            case MOMENTUM:
                return new MomentumOptimizer(layer); // ovaj mora da zna jel je 2d ili 4d weights
            case ADAGRAD:
                return new AdaGradOptimizer(layer);
            case RMSPROP:
                return new RmsPropOptimizer(layer);
            case ADADELTA:
                return new AdaDeltaOptimizer(layer);   
            case ADAM:
                return new AdamOptimizer(layer);  
            default:
                throw new DeepNettsException("Unknown optimizer:" + type);
        }
    }        
    
    public final static int ROW_IDX=0;
    public final static int COL_IDX=1;
//    public final static int DEPTH_IDX=2;
//    public final static int FOURTH_DIM_IDX=3;
    
}
