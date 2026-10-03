package deepnetts.automl;

import deepnetts.net.NeuralNetwork;
import java.util.Properties;

/**
 * Base interface for all network factories.
 * Network factory creates a neural network architecture using specified properties.
 * 
 * @param <T>  Type of network architecture that factory creates.
 */
public interface NetworkFactory<T extends NeuralNetwork> {
    
       /**
        * Creates and returns a neural network of specified type
        * @param prop properties of a neural network to create
        * @return neural network  of specified type
        */ 
       public T createNeuralNetwork(Properties prop);
       
}
