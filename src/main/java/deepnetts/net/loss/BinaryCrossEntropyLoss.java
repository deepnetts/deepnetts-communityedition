package deepnetts.net.loss;

import deepnetts.util.DeepNettsException;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.OutputLayer;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;
import java.io.Serializable;

/**
 * Cross Entropy Loss is a loss function used for binary classification tasks (two classes, single output which represents probability ).
 * It should be used in combination with sigmoid output activation function.
 * The formula: 
 * 
 * E = (1/n) * -SUM( t * ln(y) + (1-t) * ln(1-y) )
 * 
 * where t is target, and y actual output
 * Bishop, C. pg. 231, eq. 6.120
 */
public class BinaryCrossEntropyLoss implements LossFunction, Serializable {
    /**
     * http://tagkopouloslab.ucdavis.edu/?p=2197
     * When training a binary classifier, cross entropy (CE) loss is usually used as squared error loss cannot distinguish bad predictions from extremely bad predictions.
     * The CE loss is defined as follows: 
     * a z je ukupni ulaz u neuron tj. y = sigmoid(z)
     * use log-sum-exp trick
     * 
     * how tf calculates it
     * https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
     */
    
    private static final long serialVersionUID  = 3236227875327534698L;
    
    
/**
 * Cross entropy derivative is
 * 
 * dE/dy = (y-t) / y*(1-y)
 * 
 * Since denominator is same as sigmoid derivative, they are canceled when calculating delta in output layer:
 * delta = dE/dy * dy/ds = y - t
 * 
 * See 
 *     http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
 *     http://peterroelants.github.io/posts/neural_network_implementation_intermezzo01/
 * 
 */
    private TensorBase outputErrorTsr; // moze biti 1D ili 2D    
    private TensorBase patternLossTsr;    
    private float totalError;
    private float patternCount=0;    
    private float regularizationSum = 0;
    private OutputLayer outputLayer;
    private float patternLoss=0;      
    private int batchSize;
     
    public BinaryCrossEntropyLoss(NeuralNetwork neuralNet) {
        if (neuralNet.getOutputLayer().getWidth()>1) throw new DeepNettsException("BinaryCrossEntropyLoss can be only used with networks with single sigmoid output!");
        boolean isBatch = (neuralNet.getInputLayer().getTensorDim()  == 2);
        outputLayer = neuralNet.getOutputLayer();
        
        if (!isBatch) {
            outputErrorTsr = new Tensor1D(1); // ali ovo nije u batch modu - ako je u batchu dodaj 2D
        } else {
            batchSize = neuralNet.getInputLayer().getBatchSize();
            outputErrorTsr = new Tensor2D(1, batchSize);
            patternLossTsr = new Tensor1D(batchSize);

        }
    }
       
    /**
     * Calculates error for given actual and target patterns and adds that  error to total error.
     * Returns output error vector for specified actual and target outputs.
     *
     * @param predictedOutput predicted output of a neural network
     * @param targetOutput target output of a neural network
     * @return error vector for specified predicted and target outputs
     */
    /*
    @Override
    public float[] addPatternError(final float[] pred,  final float[] target) {                
       // outputError[0] = actual[0] - target[0]; // ovo je dL/dy izvod loss funkcije u odnosu na izlaz ovog neurona, kada je u outputu sigmoidna f-ja. ovo se koristi za deltu izlaznog neurona              
        // ovaj output error setuj u odgovarajuci training set itemkad g avratis
        
        // log ide u infinity kada je x == 0   
        // bce ide u infinity kada je  actual ==0 a targe == 1, 
        // ili kad je actual 1 a target 0 znaci kada je bas velika greska, kad potpuno masi - to bi trebalo resiti
        // treba popraviti i sigmoid da bi ovo radilo kako treba 
        
        // da li ga racunati ili vracati sa if-ovima? za racunanje bi mi trebao ukupni ulaz izlaznog neurona
            // ili jako blizu nuli
            // kad logaritam ode u infinity totalError postaje nana
            // najbolje uzmi input preko reference na neuralNet
//        if ((pred[0]<0.000000001) && (target[0]==1)) { // Math.log(actual[0] == infinity
//            totalError += -1;
//        } else if ((pred[0]>0.99) && (target[0]==0)) { // Math.log(1-actual[0]) == infinity
//            totalError += 1;
//        } else if ((actual[0]>0.99) && (target[0]==1)) { // Math.log(1-actual[0]) == infinity
//            totalError += 0; // this is correct no error
//        }
//        else
            //totalError += (float)(target[0] * Math.log(actual[0]) + (1-target[0]) * Math.log(1-actual[0]));
        
        // izracunavanje BCE po formuli koju koriste tensorflow i pytorch
        // https://rafayak.medium.com/how-do-tensorflow-and-keras-implement-binary-classification-and-the-binary-cross-entropy-function-e9413826da7 tf and pytorch way
        //float z = outputLayer.getSingleOutInput(); // total input of output neuron - get it somewhere
        float z = pred[0]; // p verovatnoca nisam siguran sta o dova dav - prouci detaljno i ispraci!
        // ovo z nije output nego tezinaka suma sa biasom
        //https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c
        https://rafayak.medium.com/how-do-tensorflow-and-keras-implement-binary-classification-and-the-binary-cross-entropy-function-e9413826da7
        
       // outputError[0] = (float)(Math.max(z, 0) - z * actual[0] + Math.log(1 + Math.exp(-Math.abs(z))));
        // https://datascience.stackexchange.com/questions/19819/sigmoids-stability
        outputError[0] = pred[0] - target[0]; // ovo je dL/dy izvod loss funkcije u odnosu na izlaz ovog neurona, kada je u outputu sigmoidna f-ja. ovo se koristi za deltu izlaznog neurona              
        //patternLoss = (float)(Math.max(z, 0) - z * target[0] + Math.log(1 + Math.exp(-Math.abs(z))));
       // ispravi ovo da zadrzi znak!!! ako 
       final float pred_corrected = (float)Math.max(pred[0], 1e-8f); // for numerical stability to avoid -infinity
       final float pred_corrected2 = (float)Math.max(1-pred[0], 1e-8f); // for numerical stability to avoid -infinity
       patternLoss = (float)(target[0] * Math.log(pred_corrected) + (1-target[0]) * Math.log(pred_corrected2));
     //   patternLoss = (float)(target[0] * Math.log(pred[0]) + (1-target[0]) * Math.log(1-pred[0]));
        
        totalError += patternLoss;

        patternCount++;
                
        return outputError;        
    }
    */
    @Override
    public TensorBase addPatternError(TensorBase predictedOutput, TensorBase targetOutput) {
        
        if (predictedOutput instanceof Tensor1D) {
            final float predicted = predictedOutput.getValues()[0];
            final float target = targetOutput.getValues()[0];
            final float z = predicted; // p verovatnoca nisam siguran sta o dova dav - prouci detaljno i ispraci!

            //outputError[0] = predicted - target; // ovo je dL/dy izvod loss funkcije u odnosu na izlaz ovog neurona, kada je u outputu sigmoidna f-ja. ovo se koristi za deltu izlaznog neurona              
            outputErrorTsr.getValues()[0] = predicted - target;

            final float pred_corrected =  (float) Math.max(predicted, 1e-8f); // for numerical stability to avoid -infinity
            final float pred_corrected2 = (float) Math.max(1 - predicted, 1e-8f); // for numerical stability to avoid -infinity
            patternLoss = (float) (target * Math.log(pred_corrected) + (1 - target) * Math.log(pred_corrected2));

            totalError += patternLoss;
            patternCount++;

            return outputErrorTsr;
        } else  if (predictedOutput instanceof Tensor2D) { // batch
            //outputError[0] = predicted - target; // ovo je dL/dy izvod loss funkcije u odnosu na izlaz ovog neurona, kada je u outputu sigmoidna f-ja. ovo se koristi za deltu izlaznog neurona              
            // outputErrorTsr.getValues()[0] = predicted - target;
            if (targetOutput instanceof Tensor2D) {
                predictedOutput.sub(targetOutput, outputErrorTsr);
            } else if (targetOutput instanceof Tensor4D) {
                for(int b =0; b<batchSize; b++) {
                    final float err = ((Tensor2D) predictedOutput).get(0, b) - ((Tensor4D) targetOutput).get(b, 0, 0, 0);
                    ((Tensor2D)outputErrorTsr).set(err, 0, b);
                }
            }   
            
            Tensor2D pred_corrected = (Tensor2D)predictedOutput.clone(); 
           // pred_corrected.fill(0);
            Tensor2D pred_corrected2 = (Tensor2D)predictedOutput.clone();
            //pred_corrected.fill(0);
            Tensor2D patternLossTsr = (Tensor2D)predictedOutput.clone(); // kloniranje nikako ne raditi ovde - zar nije ovaj Tensor1D
            
            pred_corrected.apply((pred)->Math.max(pred, 1e-8f)); //  mozda samo daodaj ono za znak
            pred_corrected2.apply((pred)->Math.max(1 - pred, 1e-8f));
                        
            //final float pred_corrected = (float) Math.max(predicted, 1e-8f); // for numerical stability to avoid -infinity
            //final float pred_corrected2 = (float) Math.max(1 - predicted, 1e-8f); // for numerical stability to avoid -infinity
            
            pred_corrected.apply((predc)->(float)Math.log(predc));
            pred_corrected2.apply((predc)->(float)Math.log(predc));
            // dovde je sve 100% isto mozda samo daodaj ono za znak
            
            pred_corrected.multiplyElementWise(targetOutput);
            Tensor2D targetOutputInverse = (Tensor2D)targetOutput.clone();
            targetOutputInverse.apply((v)->1-v);
            pred_corrected2.multiplyElementWise(targetOutputInverse);
            
            
           // patternLoss = (float) (target * Math.log(pred_corrected) + (1 - target) * Math.log(pred_corrected2));
            pred_corrected.addInto(pred_corrected2, patternLossTsr);
            
            totalError += patternLossTsr.sum();
            patternCount+=patternLossTsr.cols();

            return outputErrorTsr;            
        }  else if (predictedOutput instanceof Tensor4D) { // gpu batch - ovo se nikad ne izvrsava jer je output 2d
            // izracunaj razliku na izlazu mreze - isto je kao ovo gore samo sto ima druge dimenzije - view?
            final Tensor4D predictedOutput4d = (Tensor4D)predictedOutput;
            final Tensor4D targetOutput4d = (Tensor4D)targetOutput;
            
            for(int b=0; b < batchSize; b++) {           
                final float prediction = predictedOutput4d.get(b, 0, 0, 0);
                float pred_corrected =  (float) Math.max(prediction, 1e-8f);
                float pred_corrected2 =  (float) Math.max(1-prediction, 1e-8f);
                pred_corrected = (float)Math.log(pred_corrected);
                pred_corrected2 = (float)Math.log(pred_corrected2);
                
                pred_corrected = pred_corrected * targetOutput4d.get(b, 0, 0, 0);
                pred_corrected2 = pred_corrected2 * (1-targetOutput4d.get(b, 0, 0, 0));
               
                ((Tensor1D)patternLossTsr).set(pred_corrected + pred_corrected2, b); // ovo mi je cudno - patern loss u batchu
            }            
            
                                
            patternCount+=targetOutput4d.fourthDim();

            totalError += patternLossTsr.sum();

            return outputErrorTsr;               
        }else {
            throw new RuntimeException("Not supported tensor outpurt dimension");
        }
    }
    
    
    @Override
    public float getPatternLoss() {
        return patternLoss;
    }     
    
    @Override
    public void addRegularizationSum(final float regSum) {
       regularizationSum = regSum;
    }       
       
    @Override
    public float getTotal() {
        return  -totalError / patternCount;
    }
    
    @Override
    public void reset() {
        totalError = 0;
        patternCount=0;
        regularizationSum = 0;
    }


    
}