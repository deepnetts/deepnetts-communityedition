package deepnetts.net.loss;

import deepnetts.net.NeuralNetwork;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;
import java.io.Serializable;

/**
 * Average Cross Entropy Loss function commonly used for multi class classification problems.
 * 
 * E = -1/n * SUM(SUM(t*ln(y)))
 * 
 * Since its 1-of-n classification scheme, all outputs except target are zeros so it comes down to  E = -1/n * SUM(ln(y_targetIdx))
 */
public class CrossEntropyLoss implements LossFunction, Serializable {
    
    private static final long serialVersionUID  = 3646652383636911829L;
    
    
 /**
 * Sum over all outputs and training samples
 * Bishop, pg. 245, eq. 6.185
 * 
 * dE/ds = (y - t)     (when Softmax activation is used)
 * 
 * http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
 */    
    private float[] outputError; // moze biti 1D ili 2D
    private TensorBase outputErrorTsr; // moze biti 1D ili 2D
    private int targetIdx;    
    private float totalError;
    private int patternCount=0;        
    private float regularizationSum;
    private float patternLoss=0;    
    private TensorBase patternLossTsr;    
    private int batchSize;
    private Tensor1D targetIdxTsr1d=null;

    // temp
    private int batchCnt = 0;
       
    // ovaj znam kad se pravi a drugi kad se trenira izvrsava tad znam da je batch
    
    // ovaj proglasi za deprecated ili da cross entropy vezuej za layer
    public CrossEntropyLoss(NeuralNetwork neuralNet) {
        //boolean isBatch = neuralNet.getInputLayer().getHeight() > 1; // sta ako je 3d input? kapirace ga kao batch iako nije 
        boolean isBatch = ((neuralNet.getInputLayer().getTensorDim()  == 2) ||
                            (neuralNet.getInputLayer().getTensorDim()  == 4)); // ako je 2d ulaz a nije konvoluciona mreza, nego ff
        // ovaj treba da zna kako da se inicijalizje da li je batch , a to zna tek kad pocne trening a ne kad konfigurise mrezu

        // kako da napravim razliku  da li je batch ili 2d/3d ulaz
        if (!isBatch) { 
            int width = neuralNet.getOutputLayer().getWidth();
            outputError = new float[width];
            outputErrorTsr = new Tensor1D(width);
        } else {
            int width = neuralNet.getOutputLayer().getWidth();
            batchSize = neuralNet.getInputLayer().getBatchSize();
            outputError = new float[width]; // kako da napravim razliku  da li je batch ili 2d/3d ulaz nek ga kreira uvek - izbaciti
            outputErrorTsr = new Tensor2D(width, batchSize); 
            patternLossTsr = new Tensor1D(batchSize);
            targetIdxTsr1d = new Tensor1D(batchSize); 
        }
    }
               
    /**
     * Calculates and returns outpurt error vector for specified predicted and target outputs.
     * 
     * @param predictedOut predicted output from the neural network
     * @param targetOut target/desired output of the neural network
     * @return error vector for specified actual and target outputs
     */
 /*
    public float[] addPatternError(float[] predictedOut,  float[] targetOut) {        
        patternCount++;        
        
        for (int i = 0; i < predictedOut.length; i++) {                                     
            outputError[i] = predictedOut[i] - targetOut[i]; // ovo je dL/dy izvod loss funkcije u odnosu na izlaz ovog neurona - ovo se koristi za deltu izlaznog neurona
            if (targetOut[i] == 1) {                        
                targetIdx = i; // TODO: this could be set explicitly in data set in order to avoid this if     
            }
        }     
                
        patternLoss = (float)Math.log(predictedOut[targetIdx]);
        totalError += patternLoss;      
        
        return outputError;        
    }
*/
    
    @Override
    public TensorBase addPatternError(TensorBase predictedOut,  TensorBase targetOut) {              
                
        // Calculate output error dE/dy
        if (predictedOut instanceof Tensor1D) { // ovo je za obican 1d output/vektor            
            Tensor1D targetOutput1d = (Tensor1D)targetOut;
            patternCount++;
            for(int i=0; i<targetOut.numElements(); i++) { // find target idx
                if (targetOutput1d.get(i) == 1) {                        
                    targetIdx = i; // TODO: this could be set explicitly in data set in order to avoid this if     
                    break;
                }
            }            
        } else if (predictedOut instanceof Tensor2D) { // ovo je za batch gde je batch po kolonama            
            Tensor2D targetOutput2d = (Tensor2D)targetOut;
            // This is temporary fix just to unstack myself - My output target is column containing idx of the true class
            // Here I am expanding it to one-hot
            if (targetOutput2d.rows() == 1) {
                Tensor2D oneHotTarget = new Tensor2D(predictedOut.shape().getDim(0), targetOutput2d.cols());
                for (int c = 0; c < targetOutput2d.cols(); c++) {
                    int targetClass = (int)targetOutput2d.get(0, c);
                    oneHotTarget.set(1.0f, targetClass, c);
                }
                targetOut = oneHotTarget;
                targetOutput2d = oneHotTarget;
            }
            batchCnt++;

            patternCount+=targetOutput2d.cols(); // ovo zapravo dodaje broj paterna u batch-u
            // nadji koja klasa ima target idx 1 - ovo je najbolje zameniti sa int-om, odnosno indexom klase
            for(int c=0; c<targetOutput2d.cols(); c++) {
                for(int r=0; r<targetOutput2d.rows(); r++) {
                    if (targetOutput2d.get(r, c) == 1) {                        
                        targetIdxTsr1d.set(r, c);
                        break; // ne moras ici do kraja kad ga nadjes
                    }
                }                     
            }
        } else if (predictedOut instanceof Tensor4D) {
        //    throw new RuntimeException("Not calculated");
            // izracunaj razliku na izlazu mreze - isto je kao ovo gore samo sto ima druge dimenzije - view?
            Tensor4D targetOutput4d = (Tensor4D)targetOut;
            
            patternCount+=targetOutput4d.fourthDim();
            // nadji kola klasa ima target idx 1 - ovo je najbolje zameniti sa int-om, odnosno indexom klase
            for(int b=0; b<targetOutput4d.fourthDim(); b++) {
                for(int r=0; r<targetOutput4d.rows(); r++) {
                    if (targetOutput4d.get(b, 0, r, 0) == 1) {                        
                        targetIdxTsr1d.set(r, b); // sta je ovo zapravo? zapamti row index gde je target 1, b je batch
                        break; // ne moras ici do kraja kad ga nadjes
                    }
                }                     
            }            
        }
        // This can be written better - In general I do think this class needs refactor especially knowing LogSumExp and PyTorch way of impl
        predictedOut.sub(targetOut, outputErrorTsr); // ovde izracunava razliku error tensor
        
        
        // za 1d je ovo
        if (predictedOut instanceof Tensor1D) {
            final float prediction = ((Tensor1D)predictedOut).get(targetIdx);
            final float pred_corrected =  (float) Math.max(prediction, 1e-8f);
            patternLoss = (float)Math.log(pred_corrected);
            totalError += patternLoss;
        } else if (predictedOut instanceof Tensor2D) {                
            final Tensor2D predictedOutput2d = (Tensor2D)predictedOut;

            for(int c=0; c < batchSize; c++) {           
                final float prediction = predictedOutput2d.get((int)targetIdxTsr1d.get(c), c);
                final float pred_corrected =  (float) Math.max(prediction, 1e-8f);
                patternLoss = (float)Math.log(pred_corrected); 
                ((Tensor1D)patternLossTsr).set(patternLoss, c); // ovo mi je cudno - patern loss u batchu
            }
            
            totalError += patternLossTsr.sum();
        } else if (predictedOut instanceof Tensor4D) {    
            final Tensor4D predictedOutput4d = (Tensor4D)predictedOut;

            for(int b=0; b < batchSize; b++) {           
                final float prediction = predictedOutput4d.get(b, 0, (int)targetIdxTsr1d.get(b), 0);
                final float pred_corrected =  (float) Math.max(prediction, 1e-8f);
                patternLoss = (float)Math.log(pred_corrected); // ova baca NaN jer je predikcija negativna! kako to moze vrv zez alayer forward
                ((Tensor1D)patternLossTsr).set(patternLoss, b); // ovo mi je cudno - patern loss u batchu
            }
            
            totalError += patternLossTsr.sum();               
        }                   
               
        return outputErrorTsr;        
    }    
        
    @Override
    public float getPatternLoss() {
        return patternLoss;
    }    
    
    @Override
    public void addRegularizationSum(final float regSum) {
         regularizationSum = regSum; // add or set?
    }       
    
    @Override
    public float getTotal() {
        // FIX za tensor
        return  -totalError / patternCount;
    }
    
    @Override
    public void reset() {
        totalError = 0;
        patternCount=0;
        regularizationSum = 0;
    }

}
