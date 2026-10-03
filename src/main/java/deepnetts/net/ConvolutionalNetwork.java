package deepnetts.net;

import deepnetts.core.DeepNetts;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.layers.ConvolutionalLayer;
import deepnetts.net.layers.Filter;
import deepnetts.net.layers.Filters;
import deepnetts.net.layers.FlattenLayer;
import deepnetts.net.layers.FullyConnectedLayer;
import deepnetts.net.layers.InputLayer;
import deepnetts.net.layers.MaxPoolingLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.layers.SoftmaxOutputLayer;
import deepnetts.net.loss.BinaryCrossEntropyLoss;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.net.loss.LossFunction;
import deepnetts.net.loss.LossType;
import deepnetts.net.loss.MeanSquaredErrorLoss;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import deepnetts.util.RandomGenerator;
import deepnetts.tensor.TensorBase;
import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Convolutional neural network is an extension of feed forward network, which
 * can include 2D and 3D adaptive preprocessing layers (Convolutional and
 * MaxPooling layer), which is specialized to learn to recognize features in
 * images. Images are fed as 3-dimensional tensors (multidimensional arrays).
 * Although primary used for images, they can also be applied to other types of
 * problems.
 *
 * @see ConvolutionalLayer
 * @see MaxPoolingLayer
 * @see BackpropagationTrainer
 *
 * @author Zoran Sevarac
 */
public class ConvolutionalNetwork extends NeuralNetwork<BackpropagationTrainer> implements Serializable {

    private static final long serialVersionUID = 7311052836990578126L;

    private ConvolutionalNetwork() {
        super();
        setTrainer(new BackpropagationTrainer(this));
    }

    // override set input to apply image preprocessing
    @Override
    public void setInput(TensorBase input) {
        if (getPreprocessing() != null) {
            getPreprocessing().apply(input); // do this only in inference mode! i to ne iz ide nego kad ucitava sliku spolja za buffered imagr! Image set vec sve uradi sto treba
        }
        super.setInput(input);
    }

    /**
     * Returns a builder for the {@link ConvolutionalNetwork}
     *
     * @return builder instance
     */
    public static ConvolutionalNetwork.Builder builder() {
        return new Builder();
    }

    /**
     * Builder for a convolutional neural network.
     *
     * @see ConvolutionalNetwork
     */
    public static class Builder {

        private final ConvolutionalNetwork neuralNet = new ConvolutionalNetwork();

        private ActivationType defaultActivationType = ActivationType.RELU;
        private Class<CrossEntropyLoss> defaultLossFunction = CrossEntropyLoss.class;
        private boolean setDefaultActivation = false;

        /**
         * Input layer with specified width and height, and 3 channels by
         * default.
         *
         * @param width
         * @param height
         * @return
         */
        public Builder addInputLayer(int width, int height) {
            InputLayer inLayer = new InputLayer(3, width, height);
            neuralNet.setInputLayer(inLayer);
            neuralNet.addLayer(inLayer);

            return this;
        }

        /**
         * Input layer with specified width, height and number of channels
         * (depth).
         *
         * @param width width of the input Tensor
         * @param height height of the input Tensor
         * @param channels depth of the input Tensor
         * @return this builder
         */
        public Builder addInputLayer(int width, int height, int channels) {
            InputLayer inLayer = new InputLayer(channels, width, height);
            neuralNet.setInputLayer(inLayer);
            neuralNet.addLayer(inLayer);

            return this;
        }

        public Builder addInputLayer(int batchSize, int width, int height, int channels) {
            InputLayer inLayer = new InputLayer(batchSize, channels, width, height);
            neuralNet.setInputLayer(inLayer);
            neuralNet.addLayer(inLayer);

            return this;
        }

        /**
         * Adds fully connected layer with specified width and default
         * activation function. In fully connected layer each neuron is
         * connected to all outputs from previous layer.
         *
         * @param layerWidth width of the layer which corresponds to the number
         * of outputs/neurons in this layer
         * @return current builder instance
         * @see FullyConnectedLayer
         * @see ActivationType
         */
        public Builder addFullyConnectedLayer(int layerWidth) {
            FullyConnectedLayer layer = new FullyConnectedLayer(layerWidth, defaultActivationType); // ovde staviti defaultActivationType
            neuralNet.addLayer(layer);
            return this;
        }

        /**
         * Adds fully connected layer with specified width and activation
         * function. In dense layer each neuron is connected to all outputs from
         * previous layer.
         *
         * @param layerWidth width of the layer which corresponds to the number
         * of outputs/neurons in this layer
         * @param activationType type of activation function
         * @return current builder instance
         * @see FullyConnectedLayer
         * @see ActivationType
         */
        public Builder addFullyConnectedLayer(int layerWidth, ActivationType activationType) {
            FullyConnectedLayer layer = new FullyConnectedLayer(layerWidth, activationType);
            neuralNet.addLayer(layer);
            return this;
        }

        /**
         * Adds output layer to the neural network with specified width (number
         * of outputs) and layer class.
         *
         * @param layerWidth width of the layer which corresponds to number of
         * network's outputs
         * @param clazz class of the output layer
         * @return builder instance
         */
        public Builder addOutputLayer(int layerWidth, Class<? extends OutputLayer> clazz) { // ActivationType.SOFTMAX
            try {
                OutputLayer outputLayer = clazz.getDeclaredConstructor(Integer.TYPE).newInstance(layerWidth);
                neuralNet.addLayer(outputLayer);
                neuralNet.setOutputLayer(outputLayer);
            } catch (InstantiationException | IllegalAccessException | NoSuchMethodException | SecurityException | IllegalArgumentException | InvocationTargetException ex) {
                Logger.getLogger(ConvolutionalNetwork.class.getName()).log(Level.SEVERE, null, ex);
            }

            return this;
        }

        /**
         * Adds output layer to the neural network with specified width (number
         * of outputs) and activation function type.
         *
         * @param layerWidth width of the layer which corresponds to number of
         * network's outputs
         * @param activationType type of the activation function to use in
         * output layer
         * @return builder instance
         */
        public Builder addOutputLayer(int layerWidth, ActivationType activationType) {
            OutputLayer outputLayer = null;
            if (activationType.equals(ActivationType.SOFTMAX)) {
                outputLayer = new SoftmaxOutputLayer(layerWidth);
            } else {
                outputLayer = new OutputLayer(layerWidth, activationType);
                //outputLayer.setActivationType(activationType);
            }

            neuralNet.setOutputLayer(outputLayer);
            neuralNet.addLayer(outputLayer);

            return this;
        }

        /**
         * Adds a convolutional layer with the given number of
         * channels(filters), with default 3x3 filter size and default
         * activation function. Each channel(filter) is capable to learn to
         * detect a specific patern of pixels in image.
         *
         * @param channelNum number of channels(filters) in the convolutional
         * layer
         * @return builder instance
         * @see ConvolutionalLayer
         */
        public Builder addConvolutionalLayer(int channelNum) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(channelNum, Filters.ofSize(3), defaultActivationType);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        /**
         * Adds a convolutional layer with the given number of
         * channels(filters), with given activation function type and default
         * 3x3 filter size. Each channel(filter) is capable to learn to detect a
         * specific patern of pixels in image.
         *
         * @param channelNum number of channels(filters) in the convolutional
         * layer
         * @param activationType type of the activation function in the
         * convolutional layer
         * @return builder instance
         * @see ConvolutionalLayer
         * @see ActivationType
         */
        public Builder addConvolutionalLayer(int channelNum, ActivationType activationType) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(channelNum, Filters.ofSize(3), activationType);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        /**
         * Adds a convolutional layer with the given number of
         * channels(filters), with given filter size (same width and height) and
         * default activation function. Each channel(filter) is capable to learn
         * to detect a specific patern of pixels in image.
         *
         * @param channelNum number of channels(filters) in the convolutional
         * layer
         * @param filterSize size of the convolutional filter (same width and
         * height)
         * @return builder instance
         * @see ConvolutionalLayer
         */
        public Builder addConvolutionalLayer(int channelNum, int filterSize) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(channelNum, Filters.ofSize(filterSize), defaultActivationType);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        /**
         * Adds a convolutional layer with the given number of
         * channels(filters), with given filter size (same width and height) and
         * given type of activation function. Each channel(filter) is capable to
         * learn to detect a specific patern of pixels in image.
         *
         * @param channelNum number of channels(filters) in the convolutional
         * layer
         * @param filterSize size of the convolutional filter (same width and
         * height)
         * @param activationType type of the activation function in the
         * convolutional layer
         * @return builder instance
         * @see ConvolutionalLayer
         * @see ActivationType
         */
        public Builder addConvolutionalLayer(int channelNum, int filterSize, ActivationType activationType) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(channelNum, Filters.ofSize(filterSize), activationType);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        /**
         * Adds a convolutional layer with the given number of
         * channels(filters), with given width and height of convolutional
         * filter and default type of activation function. Each channel(filter)
         * is capable to learn to detect a specific patern of pixels in image.
         *
         * @param channelNum number of channels(filters) in the convolutional
         * layer
         * @param filterWidth width of a convolutional filter
         * @param filterHeight height of a convolutional filter
         * @return builder instance
         * @see ConvolutionalLayer
         */
        public Builder addConvolutionalLayer(int channelNum, int filterWidth, int filterHeight) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(channelNum, Filters.ofSize(filterWidth, filterHeight), defaultActivationType);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        /**
         * Adds a convolutional layer with the given number of
         * channels(filters), with given settings of a convolutional filter and
         * default type of activation function. Each channel(filter) is capable
         * to learn to detect a specific patern of pixels in image.
         *
         * @param channelNum number of channels(filters) in the convolutional
         * layer
         * @param filter settings of the convolutional filter(width, height,
         * stride, padding)
         * @return builder instance
         * @see ConvolutionalLayer
         */
        public Builder addConvolutionalLayer(int channelNum, Filter filter) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(channelNum, filter, defaultActivationType);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        /**
         * Adds a convolutional layer with the given width and height of
         * convolutional filters, given number of channels(filters) and
         * stride(filter step), and default type of activation function. Each
         * channel(filter) is capable to learn to detect a specific patern of
         * pixels in image.
         *
         * @param filterWidth width of a convolutional filter
         * @param filterHeight height of a convolutional filter
         * @param channelNum number of channels(filters)
         * @param stride filter stride(step)
         * @return builder instance
         * @see ConvolutionalLayer
         */
        public Builder addConvolutionalLayer(int channelNum, int filterWidth, int filterHeight, int stride) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(channelNum, filterWidth, filterHeight, stride, defaultActivationType);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        /**
         * Adds a convolutional layer with the given number of
         * channels(filters), with given settings of a convolutional filter and
         * given type of activation function. Each channel(filter) is capable to
         * learn to detect a specific patern of pixels in image.
         *
         * @param channelNum number of channels(filters) in the convolutional
         * layer
         * @param filter settings of the convolutional filter(width, height,
         * stride, padding)
         * @param activationType type of the activation function in the
         * convolutional layer
         * @return builder instance
         * @see ConvolutionalLayer
         * @see ActivationType
         */
        public Builder addConvolutionalLayer(int channelNum, Filter filter, ActivationType activationType) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(channelNum, filter, activationType);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        /**
         * Adds a convolutional layer with the given width and height of
         * convolutional filters, given number of channels(filters) and
         * stride(filter step), and given type of activation function. Each
         * channel(filter) is capable to learn to detect a specific patern of
         * pixels in image.
         *
         * @param filterWidth width of a convolutional filter
         * @param filterHeight height of a convolutional filter
         * @param channelNum number of channels(filters)
         * @param stride filter stride(step)
         * @param activationType type of the activation function in the
         * convolutional layer
         * @return builder instance
         * @see ConvolutionalLayer
         * @see ActivationType
         */
        public Builder addConvolutionalLayer(int channelNum, int filterWidth, int filterHeight, int stride, ActivationType activationType) {
            ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(channelNum, filterWidth, filterHeight, stride, activationType);
            neuralNet.addLayer(convolutionalLayer);
            return this;
        }

        /**
         * Adds a max pooling layer with given filter size and stride(filter
         * step). Max pooling layer comes after convolutional layer and reduces
         * the dimensions of the input received from the previous layer.
         * Typically filter sizes of 2 are used, which effectively halves the
         * dimensions of the input from the previous layer.
         *
         * @param filterSize size of the max pooling filter(typically 2)
         * @param stride filter step size - how many pixels will filter slide.
         * Typically same as filter size for pooling layers.
         * @return builder instance
         */
        public Builder addMaxPoolingLayer(int filterSize, int stride) {
            MaxPoolingLayer poolingLayer = new MaxPoolingLayer(filterSize, filterSize, stride);
            neuralNet.addLayer(poolingLayer);
            return this;
        }

        public Builder addMaxPoolingLayer(int filterSize) {
            MaxPoolingLayer poolingLayer = new MaxPoolingLayer(Filters.ofSize(filterSize).stride(filterSize));
            neuralNet.addLayer(poolingLayer);
            return this;
        }

        /**
         * Adds a max pooling layer with given filter size and stride(filter
         * step).Max pooling layer comes after convolutional layer and reduces
         * the dimensions of the input received from the previous
         * layer.Typically filter sizes of 2 are used, which effectively halves
         * the dimensions of the input from the previous layer.
         *
         * @param filterWidth pooling filter width
         * @param filterHeight pooling filter height
         * @param stride filter step size which is typically same as filter size
         * for pooling layers.
         * @return builder instance
         */
        public Builder addMaxPoolingLayer(int filterWidth, int filterHeight, int stride) {
            MaxPoolingLayer poolingLayer = new MaxPoolingLayer(filterWidth, filterHeight, stride);
            neuralNet.addLayer(poolingLayer);
            return this;
        }

        /**
         * Adds a max pooling layer with the given filter settings.Max pooling
         * layer comes after convolutional layer and reduces the dimensions of
         * the input received from the previous layer. Typically filter sizes of
         * 2 are used, which effectively halves the dimensions of the input from
         * the previous layer.
         *
         * @param filter settings of the pooling filter(width, height, stride,
         * padding)
         * @return builder instance
         */
        public Builder addMaxPoolingLayer(Filter filter) {
            MaxPoolingLayer poolingLayer = new MaxPoolingLayer(filter);
            neuralNet.addLayer(poolingLayer);
            return this;
        }

        /**
         * Adds a given layer to the network.
         *
         * @param layer
         * @return builder instance
         */
        public Builder addLayer(AbstractLayer layer) {
            neuralNet.addLayer(layer);
            return this;
        }

        /**
         * Sets default type of the activation function to use for all hidden
         * layers in the network.
         *
         * @param activationType type of activation function
         * @return instance of the current builder
         * @see ActivationType
         */
        public Builder hiddenActivationFunction(ActivationType activationType) {
            this.defaultActivationType = activationType;
            setDefaultActivation = true;
            return this;
        }

//        public Builder lossFunction(Class<? extends LossFunction> clazz) {
//            try {
//                LossFunction loss = clazz.getDeclaredConstructor(NeuralNetwork.class).newInstance(neuralNet); // FIX!!!
//                neuralNet.setLossFunction(loss);
//            } catch (NoSuchMethodException | SecurityException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException ex) {
//                Logger.getLogger(ConvolutionalNetwork.class.getName()).log(Level.SEVERE, null, ex);
//            }
//
//            return this;
//        }
        /**
         * Sets loss function to be used by created neural network. Loss
         * function calculates the network's error during the training as a
         * difference between actual and target output provided in training set.
         *
         * @param lossType type of a loss function
         * @return instance of the current builder
         */
        public Builder lossFunction(LossType lossType) {
            LossFunction loss = null;
            switch (lossType) {
                case MEAN_SQUARED_ERROR:
                    loss = new MeanSquaredErrorLoss(neuralNet);
                    break;
                case CROSS_ENTROPY:
                    if (neuralNet.getOutputLayer().getWidth() == 1) {
                        if (neuralNet.getOutputLayer().getActivationType() != ActivationType.SIGMOID) {
                            throw new DeepNettsException("Illegal combination of activation and loss functions (Sigmoid activation must be used with Cross Entropy Loss)");
                        }
                        loss = new BinaryCrossEntropyLoss(neuralNet);
                    } else {
                        // a sta ako je regresija ovaj uslov ne treba mada gore je case ce * iz nekog razloga sam zakomentarisao
                        //if (neuralNet.getOutputLayer().getActivationType() != ActivationType.SOFTMAX )
                        //   throw new DeepNettsException("Illegal combination of activation and loss functions (Softmax activation must be used with Cross Entropy Loss)");
                        loss = new CrossEntropyLoss(neuralNet);
                    }
                    break;
            }
            neuralNet.setLossFunction(loss);

            return this;
        }

        /**
         * Initializes random number generator with the specified seed in order
         * to get same random number sequences used for weights initialization.
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
         * Builds an instance of ConvolutionalNetwork with settings specified in
         * this builder.
         *
         * @return an instance of the ConvolutionalNetwork created by this
         * builder
         */
        public ConvolutionalNetwork build() {
            // connect and init layers, weights, etc.
            AbstractLayer prevLayer = null;

            for (int i = 0; i < neuralNet.getLayers().size(); i++) {
                AbstractLayer currentLayer = neuralNet.getLayers().get(i);
                currentLayer.setNetworkType(NetworkType.CONVOLUTIONAL);
                currentLayer.setCudaHandles(neuralNet.cudaHandles); // da li i za InputLayer? nije potrebna ali ne smeta
                 if (setDefaultActivation && !(currentLayer instanceof InputLayer) && !(currentLayer instanceof OutputLayer)) { // ne za izlazni layer
                     // on uvek setuje default actvation type to b trebalo popraviti
                    currentLayer.setActivationType(defaultActivationType); // ali ovo ne treba ovako!!! ako je vec nesto setovano onda nemoj to d agazis
                }

                // ako je ovde fc layer a prethodni je conv ili maxpooling ubaci i jedan Flatten Layer automatski, to je implementacioni detalj i on i ne mora da se vidi, sakrij ga 
                if (currentLayer instanceof FullyConnectedLayer && (prevLayer instanceof MaxPoolingLayer || prevLayer instanceof ConvolutionalLayer)) {
                    // autmatski ubaci flatten izmedju fc i conv/maxpool layera
                    AbstractLayer flattenLayer = new FlattenLayer(); // add flatten layer
                    neuralNet.getLayers().add(i, flattenLayer);
                    flattenLayer.setPrevLayer(prevLayer);
                    prevLayer.setNextlayer(flattenLayer);
                    prevLayer = flattenLayer;
                } else {
                    currentLayer.setPrevLayer(prevLayer);
                    if (prevLayer != null) { // prevLayer je null samo za input layer
                        prevLayer.setNextlayer(currentLayer);
                    }
                    prevLayer = currentLayer; // current layer becomes prev layer in next iteration
                }
            }

            // init all layers
            for (AbstractLayer layer : neuralNet.getLayers()) {
                if (DeepNetts.getInstance().useCuda()) {
                    layer.setCudaHandles(neuralNet.cudaHandles); // if use cuda
                }
                if (DeepNetts.getInstance().isMultithreaded()) {
                    layer.setThreadPool(neuralNet.threadPool);
                }
                layer.init();

            }
            // neuralNet.getLayers().forEach( layer -> layer.init() );

            // if loss is not set use default loss function
            if (neuralNet.getLossFunction() == null) {
                Builder.this.lossFunction(LossType.CROSS_ENTROPY);//defaultLossFunction
            }

            return neuralNet;
        }
    }

    /**
     * Returns weights from all layers in this network as a list of tensors.
     *
     * @return all network's weights
     */
    public List<TensorBase> getWeights() {
        List<TensorBase> weightsList = new ArrayList();
        for (AbstractLayer layer : getLayers()) {
            if (layer instanceof ConvolutionalLayer) {
                TensorBase filters = ((ConvolutionalLayer) layer).getFilters();
                weightsList.add(filters);
            } else {
                weightsList.add(layer.getWeights());
            }
        }
        return weightsList;
    }

    /**
     * Sets network's weights for all layers.
     *
     * @param weights List of weights for all layers.
     */
    public void setWeights(List<String> weights) {
        int weightsIdx = 0;

        for (int layerIdx = 1; layerIdx < getLayers().size(); layerIdx++) {
            AbstractLayer layer = getLayers().get(layerIdx);
            if (layer instanceof ConvolutionalLayer) {
                ((ConvolutionalLayer) layer).setFilters(weights.get(weightsIdx));
                weightsIdx++;
            } else if (layer instanceof FullyConnectedLayer || layer instanceof OutputLayer) {
                layer.setWeights(weights.get(weightsIdx));
                weightsIdx++;
            }
        }
    }

    /**
     * Returns delta weights for all layers. Delta weights are weight changes
     * calculated during the training procedure. Useful for debugging
     *
     * @return
     */
    public List<TensorBase> getDeltaWeights() {
        List<TensorBase> weightsList = new ArrayList();
        for (AbstractLayer layer : getLayers()) {
            weightsList.add(layer.getDeltaWeights());
        }
        return weightsList;
    }

    /**
     * Returns outputs of all layers. Useful for debugging
     *
     * @return
     */
    public List<TensorBase> getLayersOutputs() {
        List<TensorBase> outputsList = new ArrayList();
        for (AbstractLayer layer : getLayers()) {
            outputsList.add(layer.getOutputs());
        }
        return outputsList;
    }
    
    public List<TensorBase> getLayersGradients() {
        List<TensorBase> outputsList = new ArrayList();
        for (AbstractLayer layer : getLayers()) {
            outputsList.add(layer.getGradients());
        }
        return outputsList;
    }    

}
