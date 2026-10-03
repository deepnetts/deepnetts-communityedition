package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.util.DeepNettsException;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;
import java.io.Serializable;

/**
 * Implementation of ADAGRAD  {@link Optimizer} , which uses sum of squared previous gradients
 * to adjust a global learning rate for each weight.
 * Recommended initial value for the learning rate is 0.01
 * 
 * Original paper1: https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
 * 
 */
public final class AdaGradOptimizer implements Optimizer, Serializable { // better make them transient
    
    private static final long serialVersionUID = 2833280391108408107L;    
    
    // za adagrad treba koristiti manji globalni learning rate
   // https://ruder.io/optimizing-gradient-descent/
   // https://medium.com/konvergen/an-introduction-to-adagrad-f130ae871827
    // https://keras.io/api/optimizers/adagrad/
    /*
        One of Adagrad's main benefits is that it eliminates the need to manually tune the learning rate. 
        Most implementations use a default value of 0.01 and leave it at that.
    
        Adagrad's main weakness is its accumulation of the squared gradients in the denominator:
            Since every added term is positive, the accumulated sum keeps growing during training.
            This in turn causes the learning rate to shrink and eventually become infinitesimally small,
            at which point the algorithm is no longer able to acquire additional knowledge.    
    */    
    private float learningRate; // global learning rate 
    private final TensorBase prevGradSqrSum;  // ovo je Gt za gradiente za weights
    private Tensor2D prevGradSqrSum2D;  // ovo je Gt za gradiente za weights
    private Tensor4D prevGradSqrSum4D;  // ovo je Gt za gradiente za weights
    private final Tensor1D prevBiasSqrSum; // isto Gt samo za biase
    
    private float eps = 1e-7f;
             
    // sa dobrom kombinacijom lr=0.01 inicijalne akumulacije 0.1 daje odlicne rezultate z mnist
    // za 6 iteracija, 1000 slika dodje do 0.99 accuracy
    // sabiraj samo prethodne gradijente ne i tekuci u sumi gradijenata
    
    // https://keras.io/api/optimizers/adagrad/
    // https://d2l.ai/chapter_optimization/adagrad.html
    
    // One of Adagrad’s main benefits is that it eliminates the need to manually tune the learning rate. Most
    // implementations use a default value of 0:01 and leave it at that.
    
    public AdaGradOptimizer(AbstractLayer layer) {
        this.learningRate   = layer.getLearningRate();
        this.prevGradSqrSum = layer.getDeltaWeights().copy(); // pvde je problem ako je prethodni layer fltten
        this.prevBiasSqrSum = new Tensor1D(layer.getDeltaBiases().numElements());      
        
        prevGradSqrSum.fill(1f); // initial_accumulator_value - moze da bude settovana vrednost 0.1 je dobrao, a cak i 1 je dobro, kerasu je defaulta 0.1
        prevBiasSqrSum.fill(1f); // ovo mogu da budu i prvi gradijenti izracunati sa sgd
        
        if(prevGradSqrSum instanceof Tensor2D) {
            prevGradSqrSum2D = (Tensor2D)prevGradSqrSum;
        }        
        
        if(prevGradSqrSum instanceof Tensor4D) {
            prevGradSqrSum4D = (Tensor4D)prevGradSqrSum;
        }           
        
        // todo: clipnorm, clipvalue, global_clipnorm  https://keras.io/api/optimizers/adagrad/
        
    }    
    
    // ne treba da radi na max pooling. Ovim layerima i ne trba optimizer zasnovan na gradijentima
    // zasto preko conv bude null pointer za prev grad sum???
    
    // ovo racuna samo delte a promena tezina da se radi u batch-u??? a inputi?
    // cekaj a kako ide sa ulazima, inputs? u batch modu?
    // gradijent obuhvata i input!
    // deltaWeight
    // 
    @Override
    public float calculateDeltaWeight(final float grad, final int... idxs) {
            // 1. Da li je EPS ispod ili van korena  - svi ga pisu ispod! a py toch u formuli izvan
            // https://ruder.io/optimizing-gradient-descent/ ga je ubacio ispod - prouci ovo jos malo
            // i ovaj https://d2l.ai/chapter_optimization/adagrad.html kaze ispod: np.sqrt(s + eps)  eps = 1e-6
            // prema pregledanim linkovima ispod korena
            
            // 2. Da li Gt ukljucuje tekuci gradijent? Ne.https://docs.pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad

        if (idxs.length == 2) {           
            final float deltaWeight = -(learningRate / ((float) Math.sqrt(prevGradSqrSum2D.get(idxs[ROW_IDX], idxs[COL_IDX]))+eps)) * grad;
            prevGradSqrSum2D.add(grad * grad, idxs[ROW_IDX], idxs[COL_IDX]); // add sqr grad to sum of prev squared gradients                        
            return deltaWeight;
        }  else if (idxs.length==4) {                        
            final float deltaWeight = -(learningRate / ((float) Math.sqrt(prevGradSqrSum4D.get(idxs[0], idxs[1], idxs[2], idxs[3]))+eps)) * grad;           
            prevGradSqrSum4D.add(grad * grad, idxs[0], idxs[1], idxs[2], idxs[3]); // squared gradient sum                                    
            return deltaWeight;
        } else {
            throw new DeepNettsException("Bad tensor index!");
        }        
    }

    @Override
    public float calculateDeltaBias(final float grad, final int idx) { // ovo bi mogao da radi nad celim vektorom odjednom a ne pojedinacni pozivi za svaki element        
        final float deltaBias = -(learningRate / ((float) Math.sqrt(prevBiasSqrSum.get(idx))+eps)) * grad;       
        prevBiasSqrSum.add(grad * grad, idx); // squared gradient sum                 
        return deltaBias;
    }
    
    @Override
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }    

    @Override
    public TensorBase calculateDeltaWeight(TensorBase grad) {
        grad.multiply(-learningRate).div(prevGradSqrSum.copy().sqrt().add(eps));
        prevGradSqrSum.add(grad.sqr());
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    @Override
    public Tensor1D calculateDeltaBias(Tensor1D grad) {
        grad.multiply(-learningRate).div(prevBiasSqrSum.copy().sqrt().add(eps));
        prevBiasSqrSum.add(grad.sqr());        
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
    
}
