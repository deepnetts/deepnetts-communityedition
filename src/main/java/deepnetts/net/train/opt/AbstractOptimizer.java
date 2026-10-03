package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.tensor.TensorBase;


/**
 * Skeletal implementation of the Optimizer interface to
 * minimize effort to implement specific optimizers.
 */
public abstract class AbstractOptimizer  {

    protected AbstractLayer layer;
    
    protected TensorBase gradients; // gradients are salculated in layer    
    protected TensorBase deltas;
    protected TensorBase inputs;
    
    // uloga optimizaera je da uzracuna delta weight 

    protected float learningRate;   // this can also be for each weight!
   
    protected TensorBase deltaWeights;
    protected TensorBase deltaBiases;

    public AbstractOptimizer(AbstractLayer layer) {
        this.layer = layer;
        gradients = layer.getGradients();   // sve ove takodje inicijalizuj u konstruktoru
        deltas = layer.getDeltas();
        deltaWeights = layer.getDeltaWeights();
        deltaBiases = layer.getBiases();
        inputs = layer.getPrevlayer().getOutputs();
        learningRate = layer.getLearningRate();        
    }
    
    // ovo treba napraviti tako da na istu osnovu mogu lako da se dodaju optimizeri
    // kako da radi i za 2d i 3d layere
    // ovo je prakticno step 2 iz backward-a
    // FIX: ovo je izgleda nigde i ne koristi!!!
  //  @Override
//    public void optimize() { // layer treba proslediti konstruktru optimizera
//        // iteriraj neurone/delte i ulaze i za svaki ulaz izracunaj promenu tezine
//        for (int deltaRow = 0; deltaRow < deltas.getRows(); deltaRow++) {   // this iterates neurons/deltas
//            for (int inRow = 0; inRow < inputs.getRows(); inRow++) {        // iterate inputs for each neuron
//                final float grad = deltas.get(deltaRow) * inputs.get(inRow); // calculate gradient dE/dw = delta * input_of_the_realated_weight
//                gradients.set(grad, deltaRow, inRow);
//                
//                final float deltaWeight = calculate(grad);
//                deltaWeights.add(deltaWeight, deltaRow, inRow); // accumulate delta weighsts    // add or set?
//            }
//
//            final float deltaBias = calculate(deltas.get(deltaRow));
//            deltaBiases[deltaRow] += deltaBias;
//        }
//    }

    // this method implemenst specific formulas in subclasses
    public abstract float calculate(final float grad);

}
