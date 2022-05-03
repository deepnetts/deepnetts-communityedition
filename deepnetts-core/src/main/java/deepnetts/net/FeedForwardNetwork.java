/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 * This file is part of DeepNetts.
 *
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.package
 * deepnetts.core;
 */
package deepnetts.net;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.layers.FullyConnectedLayer;
import deepnetts.net.layers.InputLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.layers.SoftmaxOutputLayer;
import deepnetts.net.loss.BinaryCrossEntropyLoss;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.net.loss.LossFunction;
import deepnetts.net.loss.LossType;
import deepnetts.net.loss.MeanSquaredErrorLoss;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.List;

/**
 * Feed forward neural network architecture, also known as Multi Layer Perceptron.
 * Consists of a sequence of layers trained by Backpropagation algorithm.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public final class FeedForwardNetwork extends NeuralNetwork<BackpropagationTrainer> {

    private static final long serialVersionUID = 5819940381359274290L;    
    
    private transient Tensor inputTensor;
    

    /**
     * Private constructor allows instantiation only using builder
     */
    private FeedForwardNetwork() {
        super();
        setTrainer(new BackpropagationTrainer(this));
    }

    public void setInput(float[] inputs) {
        inputTensor.setValues(inputs); 
        setInput(inputTensor);
    }

    /**
     * Predict output for the given input
     * @param inputs
     * @return
     */
    public float[] predict(float[] inputs) {
        setInput(inputs);
        return getOutput();
    }

    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException
    {
        ois.defaultReadObject();
        
        //This has to be enabled when layers.init() methods stop touching not null class members (class fields) after deserialization.
        if (false) {
        	initClassFieldsOfNetAndAllLayers();
        }
    }
    
    /**
     * This method is called in 2 scenarios:
     * 1. Creating new ConvolutionalNetwork instance, where all class members are null and have to be initialized.
     * 2. After deserialization from saved net file, where some class members will be initialized by reading the stream during deserialization in defaultReadObject method. 
     * 
     * Init methods of all layers must be sensitive for both scenarios and check if field is null before initializing it with default new objects.
     * In most cases if the field is not null, than init method should not touch it.
     */
    private void initClassFieldsOfNetAndAllLayers() {
    	List<AbstractLayer> layers = this.getLayers();
        for (AbstractLayer cur: layers) {
            cur.init();
        }
    }    
    
    /**
     * Returns builder for Feed Forward Network
     * @return
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for FeedForwardNetwork
     */
    public static class Builder {

        /**
         * FeedForwardNetwork network that will be created and configured using
         * this builder.
         */
        private final FeedForwardNetwork network = new FeedForwardNetwork();
        private ActivationType defaultActivationType = ActivationType.TANH;
        private boolean setDefaultActivation = false;

        /**
         * Adds input addLayer with specified width to the network.
         *
         * @param width addLayer width
         * @return builder instance
         */
        public Builder addInputLayer(int width) {
            InputLayer inLayer = new InputLayer(width);
            network.addLayer(inLayer);
            network.setInputLayer(inLayer);
            network.inputTensor = new Tensor(width);

            return this;
        }

        /**
         * Adds fully connected layer with specified width and Sigmoid
         * activation function to the network.
         *
         * @param width layer width / number of neurons
         * @return builder instance
         */
        public Builder addFullyConnectedLayer(int width) {
            FullyConnectedLayer layer = new FullyConnectedLayer(width);
            network.addLayer(layer);
            return this;
        }

        public Builder addFullyConnectedLayers(int... widths) {
            for(int width : widths) {
                FullyConnectedLayer layer = new FullyConnectedLayer(width);
                network.addLayer(layer);
            }
            return this;
        }

        /**
         * Adds fully connected addLayer with specified width and activation
         * function to the network.
         *
         * @param width layer width / number of neurons
         * @param activation activation function to use for this layer
         *
         * @return builder instance
         * @see ActivationFunctions
         */
        public Builder addFullyConnectedLayer(int width, ActivationType activationType) {
            FullyConnectedLayer layer = new FullyConnectedLayer(width, activationType);
            network.addLayer(layer);
            return this;
        }

        public Builder addFullyConnectedLayers(ActivationType activationType, int... widths) {
            for(int width : widths) {
                FullyConnectedLayer layer = new FullyConnectedLayer(width, activationType);
                network.addLayer(layer);
            }
            return this;
        }

        /**
         * Adds custom layer to this network (which inherits from AbstractLayer)
         *
         * @param layer
         * @return
         */
        public Builder addLayer(AbstractLayer layer) {
            network.addLayer(layer);
            return this;
        }

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


        // hidden activation function
        public Builder hiddenActivationFunction(ActivationType activationType) {
            this.defaultActivationType = activationType;
            setDefaultActivation = true;
            return this;
        }

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
         * Initializes random number generator with specified seed in order to
         * get same random number sequences (used for weights initialization).
         *
         * @param seed
         * @return
         */
        public Builder randomSeed(long seed) {
            RandomGenerator.getDefault().initSeed(seed);
            return this;
        }

        public FeedForwardNetwork build() {

            // prodji kroz celu mrezu i inicijalizuj matrice tezina / konekcije
            // povezi sve lejere
            AbstractLayer prevLayer = null;

            // connect layers
            for (int i = 0; i < network.getLayers().size(); i++) {
                AbstractLayer layer = network.getLayers().get(i);
                if (setDefaultActivation && !(layer instanceof InputLayer) && !(layer instanceof OutputLayer)) { // ne za izlazni layer
                    layer.setActivationType(defaultActivationType); // ali ovo ne treba ovako!!! ako je vec nesto setovano onda nemoj to d agazis
                }
                layer.setPrevLayer(prevLayer);
                if (prevLayer != null) {
                    prevLayer.setNextlayer(layer);
                }
                prevLayer = layer;
            }

            // init internal layer structures (weights, outputs, deltas etc. for each layer)
            for (AbstractLayer layer : network.getLayers()) {
                layer.init();
            }

            // throw excption if loss is null - ili generalno nesto nije setovano kako treba
            return network;
        }

    }

}
