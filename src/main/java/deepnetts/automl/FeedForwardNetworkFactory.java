package deepnetts.automl;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import static deepnetts.automl.Parameters.*;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.TypedProperties;
import java.util.Properties;

/**
 * Factory for FeedForwardNetwork.
 * Creates a FeedForward neural network with specified settings.
 * It is used by AutoML to create various network architectures.
 * 
 * @see FeedForwardNetwork
 */

public class FeedForwardNetworkFactory implements NetworkFactory<FeedForwardNetwork> {
    //  kako bi bilo da ovaj uzima konkretne parameter koji se setuju
    // i da ima create metod koji setuje donamicke arameter kroz buider
    // i da se ovaj prosledjuje HyperPraamterSearchu - odlicno
    private int numInputs, numOutputs;
    private LossType lossType=LossType.MEAN_SQUARED_ERROR;
    private ActivationType hiddenActivation = ActivationType.TANH;
    private ActivationType outputActivation = ActivationType.LINEAR;
    
    // gde, kako i kada setovati ove?
    
    @Override
    public FeedForwardNetwork createNeuralNetwork(Properties params) { // prosledi samo propertije koji se menjaju kako proslediti fiksne propertije
        // imamo fiksne i varijabilne parametre
        // ali nemam jos ovde podrsku za vise skrivenih slojeva...
        
        TypedProperties tprop = new TypedProperties(params);   

        int[] hiddenLayersArr = tprop.getIntArray("hiddenLayersArr"); // svaki element sadrzi broj neurona u odgovarajucem skrivenom layeru
        
        // hidden neurons je int[] - nak budu intovi razdvojeni zarezm
        FeedForwardNetwork.Builder builder =  FeedForwardNetwork.builder()
                                                    .addInputLayer(numInputs)   // ovde treba setings za skrivene neuone
                                                    //.addFullyConnectedLayer(hiddenNeurons, hiddenActivation) // ovde ubaci hiddenLayers
                                                    .addHiddenFullyConnectedLayers(hiddenLayersArr) // ovde ubaci hiddenLayers
                                                    .addOutputLayer(numOutputs, outputActivation)
                                                    .hiddenActivationFunction(hiddenActivation)
                                                    .lossFunction(lossType); 
        
        FeedForwardNetwork network = builder.build();
        // iteriraj properties i setuj hidden neurone i backpropagation
        BackpropagationTrainer trainer = network.getTrainer();
        // a gde mu setuju lr i momentum trebalo bi ovde???
        if (tprop.containsKey(MAX_ERROR) ) // proveri kako se ponasa kada je dat samo jedan parametar
            trainer.setStopError(tprop.getFloat(MAX_ERROR));
        
        if (tprop.containsKey(STOP_EPOCHS) )
            trainer.setStopEpochs(tprop.getInt(STOP_EPOCHS));
        
        if (tprop.containsKey(LEARNING_RATE)) 
            trainer.setLearningRate(tprop.getFloat(LEARNING_RATE));        
        
        if (tprop.containsKey(OPTIMIZER))
            trainer.setOptimizer(OptimizerType.valueOf(tprop.getString(OPTIMIZER).toUpperCase()));          
               
        if (MOMENTUM.equalsIgnoreCase(tprop.getString(OPTIMIZER))) {
            if (tprop.containsKey(MOMENTUM))
                trainer.setMomentum(tprop.getFloat(MOMENTUM));
        }

         return network;
    }
    

    public void setNumInputs(int numInputs) {
        this.numInputs = numInputs;
    }

    public void setNumOutputs(int numOutputs) {
        this.numOutputs = numOutputs;
    }

    public void setLossType(LossType lossType) {
        this.lossType = lossType;
    }

    public void setHiddenActivation(ActivationType hiddenActivation) {
        this.hiddenActivation = hiddenActivation;
    }
    
    public void setOutputActivation(ActivationType outputActivation) {
        this.outputActivation = outputActivation;
    }    
    
}
