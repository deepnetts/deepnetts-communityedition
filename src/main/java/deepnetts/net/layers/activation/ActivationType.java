package deepnetts.net.layers.activation;

import deepnetts.net.layers.SoftmaxOutputLayer;

/**
 * Supported types of activation functions.
 *
 * @see ActivationFunction
 */
public enum ActivationType {
    
    /**
     * Linear activation is used in output layer for regression tasks, or just passing input forward as it is.
     */
    LINEAR,
    
    /**
     * Sigmoid activation is used in output layer for binary classification tasks and logistic regression.
     */
    SIGMOID,
    
    
    TANH,
    RELU,
    LEAKY_RELU,
    
    /**
     * SoftMax activation is typically used in output layer, for multi class classification tasks.
     * @see SoftmaxOutputLayer
     */
    SOFTMAX;   
}