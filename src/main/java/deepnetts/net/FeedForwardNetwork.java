package deepnetts.net;

import deepnetts.core.DeepNetts;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.layers.FullyConnectedLayer;
import deepnetts.net.layers.InputLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.layers.SoftmaxOutputLayer;
import deepnetts.net.layers.activation.ActivationFunction;
import deepnetts.net.loss.BinaryCrossEntropyLoss;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.net.loss.LossFunction;
import deepnetts.net.loss.LossType;
import deepnetts.net.loss.MeanSquaredErrorLoss;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import deepnetts.util.RandomGenerator;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;

/**
 * Feed forward neural network architecture, also known as Multi Layer Perceptron.
 * It consists of a sequence of neural network layers {@link deepnetts.net.layers} trained by Back-propagation {@link BackpropagationTrainer} algorithm.
 * As a minimum network must have input {@link deepnetts.net.layers.InputLayer} and output layer {@link deepnetts.net.layers.OutputLayer}.
 * For non-trivial problems it will also need several hidden fully connected layers {@link deepnetts.net.layers.FullyConnectedLayer}.
 * The easiest and recommended way to create an instance of a neural network is by using {@link deepnetts.net.FeedForwardNetwork.Builder}
 * This type of network can be used for both classification and regression tasks depending how it is configured.
 * 
 * <p>For a quick explanation about essential principles behind feed forward neural networks see the tutorial
 * <a href="https://www.deepnetts.com/blog/from-basic-machine-learning-to-deep-learning-in-5-minutes.html">From Basic Machine Learning to Deep Learning in 5 Minutes</a></p>
 * 
 * <p>For a quick overview of machine learning basics required to understand Feed Forward Network see 
 * <a href="https://www.deepnetts.com/blog/machine-learning-tutorial-for-java-developers.html">Machine Learning Tutorial for Java Developers</a></p>
 * 
 * @see FeedForwardNetwork.Builder
 * @see NeuralNetwork
 * @see BackpropagationTrainer
 * 
 */
public final class FeedForwardNetwork extends NeuralNetwork<BackpropagationTrainer> {
                                                 
    private static final long serialVersionUID = 5819940381359274290L;
        
    private transient TensorBase inputTensor; // should be transient for compatibility with community version

    /**
     * Private constructor allows instantiation only using builder.
     */
    private FeedForwardNetwork() {
        super();
        setTrainer(new BackpropagationTrainer(this));
    }

    /**
     * Sets network's input using given inputs and invokes the calculation of the network for the given input (forward pass).
     * This method is usually used only during the training.
     * You don't have to use it before the call to {@link FeedForwardNetwork#predict(float...)} since <code>predict()<code> 
     * method sets given inputs and returns calculated outputs.
     * @param inputs array of inputs to the network given as array of float values
     * @throws IllegalArgumentException if size of the input vector does not match the number of the inputs of a network
     */
    public void setInput(final float... inputs) {
        if (inputTensor.numElements() != inputs.length) {
            throw new IllegalArgumentException("Size of the input vector does not match the number of the inputs of network");
        }
        
        inputTensor.setValues(inputs); // also set size / diemnsions / shape of this vector - da li ova metoda da bude ovde??? mozda samo ff ne i zconv!
        setInput(inputTensor);
    }
    
    /**
     * Returns the network's prediction (outputs) for the given input.
     * @param inputs array of inputs to the network given as array of float values
     * @return model's prediction as network's output
     * @throws IllegalArgumentException if size of the input vector does not match the number of the inputs of a network
     */
    public float[] predict(float... inputs) {
        if (inputTensor.numElements() != inputs.length) {
            throw new IllegalArgumentException("Size of the input vector does not match the number of the inputs of a network");
        }
        
        setInput(Tensor1D.of(inputs));
        return getOutputAsTensor().getValues();
    }      
    
    /**
     * Returns network output for the given input.
     * This method is deprecated and predict method should be used instead.
     * @param inputs 
     * @return
     * @deprecated 
     */
    @Deprecated
    public float[] getOutput(final float[] inputs) {
        setInput(inputs);
        return getOutputAsTensor().getValues();
    }


    /**
     * Returns a builder for the {@link FeedForwardNetwork}
     * @return builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder of a {@link FeedForwardNetwork} instance.
     * Provides methods for setting all the components of a feed forward neural network and 
     * performs basic validation of settings in order to prevent illegal configuration.
     * 
     * @see FeedForwardNetwork
     */
    public static class Builder {

        /**
         * FeedForwardNetwork network that will be created and configured using
         * this builder.
         */
        private final FeedForwardNetwork network = new FeedForwardNetwork();
        
        /**
         * Default activation type for the hidden layers
         */
        private ActivationType defaultActivationType = ActivationType.RELU;
        
        /**
         * Flag to indicate if defaultActivationType should be used by the builder.
         */
        private boolean setDefaultActivation = false;

        /**
         * Adds input layer with the specified layerWidth (number of inputs) to the network.
         * Input layer is always the first layer in the network which accepts the external input.
         * 
         * @param layerWidth width of the input layer that corresponds to the number of network's inputs
         * @return builder instance
         */
        public Builder addInputLayer(int layerWidth) {
            if (!network.getLayers().isEmpty()) throw new DeepNettsException("Input layer must be the first layer in the network!");
            InputLayer inLayer = new InputLayer(layerWidth);            
            network.addLayer(inLayer);
            network.setInputLayer(inLayer);
            network.inputTensor = new Tensor1D(layerWidth);

            return this;
        }
        
        public Builder addInputLayer(int layerWidth, int batchSize) {
            if (!network.getLayers().isEmpty()) throw new DeepNettsException("Input layer must be the first layer in the network!");
            InputLayer inLayer = new InputLayer(layerWidth, batchSize);            
            network.addLayer(inLayer);
            network.setInputLayer(inLayer);
            network.inputTensor = new Tensor2D(layerWidth, batchSize);

            return this;
        }        
        
        public Builder addInputLayer(int layerWidth, int batchSize, boolean isGpu) {
            if (!network.getLayers().isEmpty()) throw new DeepNettsException("Input layer must be the first layer in the network!");
            InputLayer inLayer = new InputLayer(layerWidth, batchSize, isGpu);            
            network.addLayer(inLayer);
            network.setInputLayer(inLayer);
            network.inputTensor = new Tensor4D(batchSize, 1, layerWidth, 1);

            return this;
        }           

        /**
         * Adds to the network a fully connected layer with specified width and Relu activation function by default.
         *
         * @param layerWidth  width of the layer / number of neurons
         * @return builder instance
         * @see FullyConnectedLayer
         */
        public Builder addFullyConnectedLayer(int layerWidth) {
            FullyConnectedLayer layer = new FullyConnectedLayer(layerWidth, defaultActivationType);
            network.addLayer(layer);
            return this;
        }

        /**
         * Adds to the network several hidden fully connected layers with specified widths and default hidden activation function by default.
         * @param layerWidths an array with widths for hidden fully connected layers.
         * @return builder instance
         * @see FullyConnectedLayer
         */
        public Builder addHiddenFullyConnectedLayers(int... layerWidths) {
            for(int width : layerWidths) {
                FullyConnectedLayer layer = new FullyConnectedLayer(width, defaultActivationType);
                network.addLayer(layer);
            }
            return this;
        }

        /**
         * Adds fully connected addLayer with specified width and activation
         * function to the network.
         *
         * @param layerWidth width of the layer to add
         * @param activationType type of the activation function for layer to add
         *
         * @return builder instance
         * @see ActivationFunction
         */
        public Builder addFullyConnectedLayer(int layerWidth, ActivationType activationType) {
            FullyConnectedLayer layer = new FullyConnectedLayer(layerWidth, activationType);
            network.addLayer(layer);
            return this;
        }

        /**
         * Adds fully connected hidden layers with widths given in layerWidths param and given activation function type.
         * @param activationType type of activation function in hidden layers
         * @param layerWidths widths of the hidden layers
         * @return 
         */
        public Builder addHiddenFullyConnectedLayers(ActivationType activationType, int... layerWidths) {
            for(int width : layerWidths) {
                FullyConnectedLayer layer = new FullyConnectedLayer(width, activationType);
                network.addLayer(layer);
            }
            return this;
        }

        /**
         * Adds custom layer to this network (which inherits from AbstractLayer)
         *
         * @param layer
         * @return builder instance
         */
        public Builder addLayer(AbstractLayer layer) {
            network.addLayer(layer);
            return this;
        }

        /**
         * Adds output layer to the neural network with specified width (number of outputs) and activation function type.
         * @param width layer with which corresponds to number of network's outputs
         * @param activationType type of the activation function to use in output layer
         * @return builder instance
         */
        public Builder addOutputLayer(int width, ActivationType activationType) {
            OutputLayer outputLayer = null;
            if (activationType.equals(ActivationType.SOFTMAX)) {
                outputLayer = new SoftmaxOutputLayer(width);
            } else {
                outputLayer = new OutputLayer(width, activationType);
            }

            network.setOutputLayer(outputLayer);
            network.addLayer(outputLayer);

            return this;
        }
        

        /**
         * Sets default type of the activation function to use for all hidden layers in the network.
         * @param activationType type of activation function
         * @return instance of the current builder
         * @see ActivationType
         */
        public Builder hiddenActivationFunction(ActivationType activationType) {
            this.defaultActivationType = activationType;
            setDefaultActivation = true;
            return this;
        }


        /**
         * Sets loss function to be used by created neural network.
         * Loss function calculates the network's error during the training as 
         * a difference between actual and target output provided in training set.
         * 
         * @param lossType type of a loss function
         * @return instance of the current builder
         */
        public Builder lossFunction(LossType lossType) {
            LossFunction loss = null;
            switch (lossType) {
                case MEAN_SQUARED_ERROR:
                    loss = new MeanSquaredErrorLoss(network);
                    break;
                case CROSS_ENTROPY:
                    if (network.getOutputLayer().getWidth() == 1) {
                        loss = new BinaryCrossEntropyLoss(network);
                    } else {
                        loss = new CrossEntropyLoss(network);
                    }
                    break;
            }
            network.setLossFunction(loss);
            return this;
        }

        /**
         * Initializes random number generator with the specified seed in order to
         * get same random number sequences used for weights initialization.
         * Specifying this value enables getting same/repeatable initialization.
         *
         * @param seed
         * @return instance of the current builder
         */
        public Builder randomSeed(long seed) {
            RandomGenerator.getDefault().initSeed(seed);
            return this;
        }

        /**
         * Builds an instance of FeedForwardNetwork with settings specified in this builder.
         * @return an instance of the FeedForwardNetwork created by this builder
         */
        public FeedForwardNetwork build() {

            // prodji kroz celu mrezu i inicijalizuj matrice tezina / konekcije
            // povezi sve lejere
            AbstractLayer prevLayer = null;

            // connect layers
            for (int i = 0; i < network.getLayers().size(); i++) {
                AbstractLayer layer = network.getLayers().get(i);
                layer.setPrevLayer(prevLayer);
                layer.setNetworkType(NetworkType.FEEDFORWARD);
                if (prevLayer != null) {
                    prevLayer.setNextlayer(layer);
                }
                prevLayer = layer;
            }

                        
            // init internal layer structures (weights, outputs, deltas etc. for each layer)
            for (AbstractLayer layer : network.getLayers()) {
                if (DeepNetts.getInstance().useCuda()) {
                    layer.setCudaHandles(network.cudaHandles); // if use cuda                    
                }
                if (DeepNetts.getInstance().isMultithreaded()) {
                    layer.setThreadPool(network.threadPool);
                }
                                
                layer.init();
            }

            // throw excption if loss is null - ili generalno nesto nije setovano kako treba
            return network;
        }

    }

}
