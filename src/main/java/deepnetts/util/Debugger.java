
package deepnetts.util;

import deepnetts.core.DeepNetts;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.Layer;
import java.util.logging.Logger;

public class Debugger {
    
    static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());
   
    public static void printOutputs(NeuralNetwork<?> nnet) {
        LOGGER.info("Outputs");
        for(Layer layer : nnet.getLayers()) {
            LOGGER.info(layer.getOutputs().toString());
        }
    }
    
    
    public static void printWeights(NeuralNetwork<?> nnet) {
        LOGGER.info("Weights");
        for(AbstractLayer layer : nnet.getLayers()) {
            LOGGER.info(layer.getWeights().toString());
        }
    }    
}
