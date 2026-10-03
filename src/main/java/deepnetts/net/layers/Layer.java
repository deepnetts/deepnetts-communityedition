package deepnetts.net.layers;

import deepnetts.tensor.Tensor;

/**
 * Common base interface for all types of neural network layers.
 * Layer is a basic building block of a neural network.
 * Neural network typically consists of a sequence of layers.
 * Layer performs mathematical operations on it's inputs, and provides output which is used as an input for the next layer.
 *
 * @param <O> specific type of output tensor
 * @see AbstractLayer
 * @see LayerType
 * @see deepnetts.net.layers
 */
public interface Layer<O extends Tensor> extends Forward, Backward {

    /**
     * Performs calculation of layer outputs in forward pass of a neural network .
     */
    @Override
    public void forward();

    /**
     * Performs weight parameters adjustment in backward pass during training of a neural network.
     */
    @Override
    public void backward();

    /**
     * Returns output of this layer (as a tensor).
     * @return layer output as a tensor
     */
    public O getOutputs();

    /**
     * Returns layer deltas/errors (as a tensor).
     * Deltas are accumulated errors propagated from the next layer.
     * @return layer deltas tensor
     */
    public O getDeltas();

}