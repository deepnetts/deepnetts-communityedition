package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.train.TrainingEvent;
import deepnetts.net.train.TrainingListener;
import deepnetts.util.DeepNettsException;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;
import java.io.Serializable;

/**
 * Implementation of Adam optimizer which uses an estimation of a gradient statistic (mean and variance) to adjust learning rate for each weight.
 * Adaptive Moment Estimation (Adam) is a method that computes adaptive learning rates for each parameter/weight.
 * 
 * Recommended(default) settings:
 *      learningRate=0.001
 *      epsilon=1e-8 small constant to prevent division by zero / ensure numerical stability
 *      beta1=0.9   exponential decay rate for the first moment estimate(mean) 
 *      beta2=0.999 exponential decay rate for the second moment estimate(mean)
 *      biasCorrection1=0.9 correct exp avg in initial iterations when it is 0
 *      biasCorrection2=0.999 correct exp avg in initial iterations when it is 0
 * 
 * FORMULAS:
 * 
 *      g(t) is a gradient at step t
 * 
 *      m(t)=beta1*m(t-1) + (1-beta1)*g(t)  estimate of a gradient mean using exponential moving average of gradients
 *      v(t)=beta2*v(t-1) + (1-beta2)*g(t)^2  estimate of a gradient variance using exponential moving average of grad squares
 * 
 *                          -learningRate * m(t)                
 *      deltaWeight(t) = __________________________   
 *                          sqrt(v(t))+epsilon
 *
 * 
 * Paper: ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION https://arxiv.org/abs/1412.6980
 * 
 */

// on ga prakticno centrira/normalizuje/standardizuje, oduzima mean i deli sa std!
public final class AdamOptimizer implements Serializable, Optimizer, TrainingListener {
   /*
    https://machinelearningmastery.com/adam-optimization-from-scratch/
 * za learning rate koristiti 0.001f i ond akonvergira inace baca NaN odmah
 * 
 * Adaptive Moment Estimation (Adam) [14] is another method that computes adaptive learning rates for each parameter. 
 * In addition to storing an exponentially decaying average of past squared gradients vt like Adadelta and RMSprop,
 * Adam also keeps an exponentially decaying average of past gradients mt, similar to momentum.
 * 
 *  adam pored sume kvdrata gradijenata cuva i sumu gradijeneata
    
    izbaci bias correction faktor  i prvu iteraciju uradi sa standardnim sgd

    */
    
    private static final long serialVersionUID = 6890113339772540541L;    
    
    private float learningRate; // global learning rate 
    
    // moving average sum of sqared gradients
    private final TensorBase movAvgGradSqrSum;  
    private Tensor2D movAvgGradSqrSum2D;  
    private Tensor4D movAvgGradSqrSum4D;  
    private final Tensor1D movAvgBiasGradSqrSum;
    
    // moving average of sradient sum
    private final  TensorBase movAvgGradSum;  // add prev grad sum not only sqr sum
    private  Tensor2D movAvgGradSum2D;
    private  Tensor4D movAvgGradSum4D;
    private final  Tensor1D movAvgBiasGradSum;
    
    private final float beta1=0.9f;
    private final float beta2=0.999f;
    
    // bias correction factor for mov avg in the beggining, since it is zero
    private float biasCorrection1=0.9f;        
    private float biasCorrection2=0.999f;  
    
    private float eps = 1e-8f; // in original paper e-8 in keras e-7    
        
    
    public AdamOptimizer(AbstractLayer layer) {
        this.learningRate = layer.getLearningRate();

        this.movAvgBiasGradSqrSum = new Tensor1D(layer.getDeltaBiases().numElements());                
        this.movAvgGradSqrSum = layer.getDeltaWeights().copy();   // mov avg of squared grad sum
        if(movAvgGradSqrSum instanceof Tensor2D) {
            movAvgGradSqrSum2D = (Tensor2D)movAvgGradSqrSum;
        } else if(movAvgGradSqrSum instanceof Tensor4D) {
            movAvgGradSqrSum4D = (Tensor4D)movAvgGradSqrSum;
        }   
        
        this.movAvgBiasGradSum = new Tensor1D(layer.getDeltaBiases().numElements());              
        this.movAvgGradSum = layer.getDeltaWeights().copy();   // mov avg of grad sum
        if(movAvgGradSum instanceof Tensor2D) {
            movAvgGradSum2D = (Tensor2D)movAvgGradSum;
        } else if(movAvgGradSqrSum instanceof Tensor4D) {
            movAvgGradSum4D = (Tensor4D)movAvgGradSum;
        }      
        
    }    
    
    // ave izracunaj kao u AdaDelta i dodaj deo za sumu gradijenata
    // dodaj i korkciju ibasa - procitaj detaljnije tu rkoreciju biasa i zasto
    // https://medium.com/@weidagang/demystifying-the-adam-optimizer-in-machine-learning-4401d162cb9e
    // https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for
    // https://medium.com/@weidagang/demystifying-the-adam-optimizer-in-machine-learning-4401d162cb9e
    // https://www.geeksforgeeks.org/adam-optimizer/
     // https://ruder.io/optimizing-gradient-descent/index.html#adam
    @Override
    public float calculateDeltaWeight(float grad, final int... idxs) {
        if (Math.abs(grad) > 0.9f) grad = Math.signum(grad)*0.9f; // limit gradients
        if (idxs.length == 2) {
            
            // calculate moving average gradient sum
            final float movAvgGrSum = beta1 * movAvgGradSum2D.get(idxs[ROW_IDX], idxs[COL_IDX]) + (1-beta1) * grad;
            movAvgGradSum2D.set(movAvgGrSum, idxs[ROW_IDX], idxs[COL_IDX]);
            final float movAvgGrSumCorrected = movAvgGrSum / (1-biasCorrection1);
            
            // calculate moving average gradient sqr sum
            final float movAvgGrSqrSum = beta2 * movAvgGradSqrSum2D.get(idxs[ROW_IDX], idxs[COL_IDX]) + (1-beta2) * grad * grad;
            movAvgGradSqrSum2D.set(movAvgGrSqrSum, idxs[ROW_IDX], idxs[COL_IDX]);
            final float movAvgGrSqrSumCorrected = movAvgGrSqrSum / (1-biasCorrection2);
       
             // calculate weight change
            final float deltaWeight = -(learningRate * movAvgGrSumCorrected) / ((float)Math.sqrt(movAvgGrSqrSumCorrected)+eps);
           
            return deltaWeight;
        }  else if (idxs.length==4) {        
            
            // calculate moving average gradient sum
            final float movAvgGrSum = beta1 * movAvgGradSum4D.get(idxs[0], idxs[1], idxs[2], idxs[3]) + (1-beta1) * grad;
            movAvgGradSum4D.set(movAvgGrSum, idxs[0], idxs[1], idxs[2], idxs[3]);
            final float movAvgGrSumCorrected = movAvgGrSum / (1-biasCorrection1);

            // calculate moving average gradient sqr sum
            final float movAvgGrSqrSum = beta2 * movAvgGradSqrSum4D.get(idxs[0], idxs[1], idxs[2], idxs[3]) + (1-beta2) * grad * grad;
            movAvgGradSqrSum4D.set(movAvgGrSqrSum, idxs[0], idxs[1], idxs[2], idxs[3]);
            final float movAvgGrSqrSumCorrected = movAvgGrSqrSum / (1-biasCorrection2);

            // calculate weight change
            final float deltaWeight = -(learningRate * movAvgGrSumCorrected) / ((float)Math.sqrt(movAvgGrSqrSumCorrected)+eps);
            
            return deltaWeight;
        } else {
            throw new DeepNettsException("Bad tensor index!");
        }        
    }

    @Override
    public float calculateDeltaBias(float grad, int idx) {
        if (Math.abs(grad) > 0.9f) grad = Math.signum(grad)*0.9f; // limit gradients
        
         // calculate moving average bias gradient sum
        final float movAvgBiasGrSum = beta1 * movAvgBiasGradSum.get(idx) + (1-beta1) * grad;
        movAvgBiasGradSum.set(movAvgBiasGrSum, idx);  
        final float movAvgGrSumCorrected = movAvgBiasGrSum / (1-biasCorrection1);
        
        // calculate moving average gradient sqr sum
        final float movAvgSqrSum = beta2 * movAvgBiasGradSqrSum.get(idx) + (1-beta2) * grad * grad;
        movAvgBiasGradSqrSum.set(movAvgSqrSum, idx);         
        final float movAvgBiasGrSqrSumCorrected = movAvgSqrSum / (1-biasCorrection2);              
        
        final float deltaBias = -(float)(learningRate*movAvgGrSumCorrected) / ((float)Math.sqrt(movAvgBiasGrSqrSumCorrected)+eps);  
       
        return deltaBias;
    }
    
    @Override
    public void handleEvent(TrainingEvent event) {
        if (event.getType() == TrainingEvent.EPOCH_STARTED) {
            // TODO: akoporaste preko 1 korekcija ide u negativno
            biasCorrection1 = biasCorrection1 * beta1; 
            biasCorrection2 = biasCorrection2 * beta2;
        }
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
