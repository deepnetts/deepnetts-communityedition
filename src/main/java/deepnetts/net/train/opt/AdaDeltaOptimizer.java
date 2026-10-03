package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.ConvolutionalLayer;
import deepnetts.net.layers.MaxPoolingLayer;
import deepnetts.net.train.TrainingEvent;
import deepnetts.net.train.TrainingListener;
import deepnetts.util.DeepNettsException;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;

/**
 * Implementation of ADADELTA which is a modification of AdaGrad that uses only a limited window of previous gradients.
 * Warning: Implementation of this optimizer is experimental and still unstable.
 *
 * @see Optimizer
 * @see AdaGradOptimizer
 * @see <a href="https://arxiv.org/pdf/1212.5701">ADADELTA Paper</a>
 */
public final class AdaDeltaOptimizer implements Optimizer, TrainingListener {
   /*
 * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 * 
 * https://ruder.io/optimizing-gradient-descent/index.html#fnref13
 *
 * 
 * Adadelta [13] is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.
 * Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to
 * some fixed size w.
 * 
 * Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. Retrieved from http://arxiv.org/abs/1212.5701
 *     
 */
    private static final long serialVersionUID = 5323161873591333899L;    
    
    private final TensorBase movAvgGradSqrSum;  
    private Tensor2D movAvgGradSqrSum2D;  
    private Tensor4D movAvgGradSqrSum4D;  
    private final Tensor1D movAvgBiasSqrSum;
    private TensorBase movAvgDeltaWeightsSqrSum; 
    private Tensor2D movAvgDeltaWeightsSqrSum2D; 
    private Tensor4D movAvgDeltaWeightsSqrSum4D; 
    private Tensor1D movAvgDeltaBiasSum; 
    
    private float learningRate;
    private float rho = 0.9f;
    private float rho2 = 0.999f;
    
    private float eps = 1e-8f;
    
    private float biasCorrection1=0.9f;        
    private float biasCorrection2=0.999f;      
    
    float initialAcc = 0.0f;
    
    // koliko je rho? kolika je pocetna akumulacija , kad nema informacija?
    // TODO: proveri sve formule
    // povecaj pocetni gradijent, nek ga izracuna sa sgd ili uradi bias correction /0.1 je kao *10
    
    // https://datascience.stackexchange.com/questions/121790/what-exactly-is-gradient-norm
    public AdaDeltaOptimizer(AbstractLayer layer) {

        this.movAvgGradSqrSum = (TensorBase)layer.getDeltaWeights().copy();// moving average of sqyared gradient sum
        movAvgGradSqrSum.fill(initialAcc);
        this.movAvgBiasSqrSum = new Tensor1D(layer.getDeltaBiases().numElements());            
        movAvgBiasSqrSum.fill(initialAcc);
        
        // cini mi se da s ei ovo dole unifikovalo
        if (layer instanceof ConvolutionalLayer) {
            movAvgDeltaWeightsSqrSum = (Tensor4D)layer.getDeltaWeights().copy(); // this returns filters
            movAvgDeltaWeightsSqrSum.fill(initialAcc); // znaci napunio ga je learning ratom
            movAvgDeltaBiasSum = new Tensor1D(layer.getDeltaBiases().numElements());
            movAvgDeltaBiasSum.fill(initialAcc);                       
        } else if (!(layer instanceof MaxPoolingLayer)) {
            movAvgDeltaWeightsSqrSum = (TensorBase)layer.getDeltaWeights().copy();
            movAvgDeltaWeightsSqrSum.fill(initialAcc);
            movAvgDeltaBiasSum = new Tensor1D(layer.getDeltaBiases().numElements());
            movAvgDeltaBiasSum.fill(initialAcc);        
        }
        
        if (movAvgGradSqrSum instanceof Tensor2D) {
            movAvgGradSqrSum2D = (Tensor2D) movAvgGradSqrSum;
            movAvgDeltaWeightsSqrSum2D = (Tensor2D) movAvgDeltaWeightsSqrSum; // /evo ga zajeb!!!!!
        }

        if (movAvgGradSqrSum instanceof Tensor4D) {
            movAvgGradSqrSum4D = (Tensor4D) movAvgGradSqrSum;
            movAvgGradSqrSum4D.fill(initialAcc);
            movAvgDeltaWeightsSqrSum4D = (Tensor4D) movAvgDeltaWeightsSqrSum;
            movAvgDeltaWeightsSqrSum4D.fill(initialAcc);
        }      
    }    
      
    
    // ovo treba da racuna samo delte a promena tezina da se radi u batch-u
    // treba izracunati moven average za delta weights i to staviti u brojic umesto  learning rate-a
    // final float movAvgGrSum = 0.9f * deltaWeights.get(idxs[ROW_IDX], idxs[COL_IDX]) + 0.1f * deltaWeight * deltaWeight
    // a ova metoda zapravo racuna delta weight
    // a mozes da psistupas starima preko layera ako treba
    // ovo ispod cak deluje dobro
    @Override
    public float calculateDeltaWeight(float grad, final int... idxs) { 
        if (Math.abs(grad) > 0.9f) grad = Math.signum(grad)*0.9f; // 
        if (idxs.length == 2) {
            float movAvgGrSqrSum = rho * movAvgGradSqrSum2D.get(idxs[ROW_IDX], idxs[COL_IDX]) + (1-rho) * grad * grad;
            movAvgGradSqrSum2D.set(movAvgGrSqrSum, idxs[ROW_IDX], idxs[COL_IDX]);      
            
            final float movAvgGrSqrSumCorrected = movAvgGrSqrSum / (1-biasCorrection1);            
            final float movAvgDwSqrSumCorrected = movAvgDeltaWeightsSqrSum2D.get(idxs[ROW_IDX], idxs[COL_IDX]) / (1-biasCorrection2);

            /*  where the same constant EPS is added to the numerator RMS as well.
                This constant serves the purpose both to start off the first
                 iteration where dx0 = 0 and to ensure progress continues to be made even if previous updates become small. */
            
            // ovde treba primeniti bias correction: https://ruder.io/optimizing-gradient-descent/index.html#adam
            // sta ce mi ovo bias correction????
            // https://d2l.ai/chapter_optimization/adadelta.html evo ovde lepo
//            final float deltaWeight = -(float)(Math.sqrt(movAvgDeltaWeightsSqrSum2D.get(idxs[ROW_IDX], idxs[COL_IDX])+ eps)  / 
//                                               Math.sqrt(movAvgGradSqrSum2D.get(idxs[ROW_IDX], idxs[COL_IDX])+ eps)) 
//                                               * grad;

//            final float deltaWeight = -(float)(Math.sqrt(movAvgDwSqrSumCorrected+ eps)  / 
//                                               Math.sqrt(movAvgGrSqrSumCorrected+ eps)) 
//                                               * grad;
   
            float adjustedLearningRate = (float)( Math.sqrt(movAvgDwSqrSumCorrected + eps) / Math.sqrt(movAvgGrSqrSumCorrected+eps));
            if (adjustedLearningRate >0.01f) adjustedLearningRate = 0.01f;
            final float deltaWeight = - adjustedLearningRate * grad;
 
            final float movAvgDwSum = rho2 * movAvgDeltaWeightsSqrSum2D.get(idxs[ROW_IDX], idxs[COL_IDX]) + (1-rho2) * deltaWeight * deltaWeight;
            movAvgDeltaWeightsSqrSum2D.set(movAvgDwSum, idxs[ROW_IDX], idxs[COL_IDX]);
              
//            if (grad!=0) {
//                System.out.println("grad: "+grad+" movAvgGrSqrSumCorrected: "+movAvgGrSqrSumCorrected + " movAvgDwSqrSumCorrected: "+movAvgDwSqrSumCorrected+" adjustedLearningRate:"+adjustedLearningRate+" deltaWeight:"+ deltaWeight);
//            }
            
            return deltaWeight;
        }  else if (idxs.length==4) { // redosled i sume moradju da se doteraju prvi put sgd ova grda suma bi isto trebala ispod

            float movAvgGrSqrSum = rho * movAvgGradSqrSum4D.get(idxs[0], idxs[1], idxs[2], idxs[3]) + (1-rho) * grad * grad;            
            movAvgGradSqrSum4D.set(movAvgGrSqrSum, idxs[0], idxs[1], idxs[2], idxs[3]);                          
            final float movAvgGrSqrSumCorrected = movAvgGrSqrSum / (1-biasCorrection1);            
            final float movAvgDwSqrSumCorrected = movAvgDeltaWeightsSqrSum2D.get(idxs[ROW_IDX], idxs[COL_IDX]) / (1-biasCorrection2);
            
            float adjustedLearningRate = (float)( Math.sqrt(movAvgDwSqrSumCorrected + eps) / Math.sqrt(movAvgGrSqrSumCorrected+eps));
            if (adjustedLearningRate >0.01f) adjustedLearningRate = 0.01f;
            final float deltaWeight = - adjustedLearningRate * grad;
                 
            final float movAvgDwSum = rho2 * movAvgDeltaWeightsSqrSum4D.get(idxs[0], idxs[1], idxs[2], idxs[3]) + (1-rho2) * deltaWeight * deltaWeight;
            movAvgDeltaWeightsSqrSum4D.set(movAvgDwSum, idxs[0], idxs[1], idxs[2], idxs[3]);
       
//            if (grad!=0) {
//                System.out.println("grad: "+grad+" movAvgGrSqrSumCorrected: "+movAvgGrSqrSumCorrected + " movAvgDwSqrSumCorrected: "+movAvgDwSqrSumCorrected+" adjustedLearningRate:"+adjustedLearningRate+" deltaWeight:"+ deltaWeight);
//            }
            return deltaWeight;
        } else {
            throw new DeepNettsException("Bad tensor index!");
        }        
    }

    @Override
    public float calculateDeltaBias(float grad, int idx) {    
        if (Math.abs(grad) > 0.9f) grad = Math.signum(grad)*0.9f;
        
        float movAvgBiasGradSqrSum = rho * movAvgBiasSqrSum.get(idx) + (1-rho) * grad * grad;
        movAvgBiasSqrSum.set(movAvgBiasGradSqrSum, idx);    
        final float movAvgGrSqrSumCorrected = movAvgBiasGradSqrSum / (1-biasCorrection1);
        final float movAvgDwSqrSumCorrected = movAvgDeltaBiasSum.get(idx) / (1-biasCorrection2);        
        
//        final float deltaBias = -((float)Math.sqrt(movAvgDwSqrSumCorrected + eps) /
//                                  ((float) Math.sqrt(movAvgGrSqrSumCorrected+eps))) * grad;
                           
            float adjustedLearningRate = (float)( Math.sqrt(movAvgDwSqrSumCorrected + eps) / Math.sqrt(movAvgGrSqrSumCorrected+eps));
            if (adjustedLearningRate >0.01f) adjustedLearningRate = 0.01f;
            
            final float deltaBias = - adjustedLearningRate * grad;        
        
        final float movAvgSqrDeltaBias = rho2 * movAvgDeltaBiasSum.get(idx) + (1-rho2) * deltaBias * deltaBias;
        movAvgDeltaBiasSum.set(movAvgSqrDeltaBias, idx);
        
//            if (grad!=0) {
//                System.out.println("grad: "+grad+" movAvgGrSqrSumCorrected: "+movAvgGrSqrSumCorrected + " movAvgDwSqrSumCorrected: "+movAvgDwSqrSumCorrected+" adjustedLearningRate:"+adjustedLearningRate+" deltaBias:"+ deltaBias);
//            }
         //final float   deltaBias=0f;
         return deltaBias;
    }

    // https://ruder.io/optimizing-gradient-descent/index.html#adam
    @Override
    public void handleEvent(TrainingEvent event) {
        if (event.getType() == TrainingEvent.EPOCH_STARTED) {
            // limit it so it cannon grow larger then 1
            if (biasCorrection1 < 0.9f)
                biasCorrection1 = biasCorrection1 * 0.9f; 
            
            if (biasCorrection2 < 0.9f)
                biasCorrection2 = biasCorrection2 * 0.999f;
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
