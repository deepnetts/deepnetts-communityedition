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
package deepnetts.net.layers;

import deepnetts.net.layers.activation.ActivationFunctions;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.core.DeepNetts;
import deepnetts.net.train.opt.Optimizers;
import deepnetts.util.WeightsInit;
import deepnetts.util.Tensor;
import java.util.Arrays;
import java.util.logging.Logger;
import deepnetts.net.layers.activation.ActivationFunction;
import deepnetts.util.DeepNettsException;

/**
 * Fully connected layer is used as hidden layer in the neural network, and it
 * has a single row of units/nodes/neurons connected to all neurons in
 * previous and next layer.
 * Previous connected layer can be input fully connected, convolutional or max pooling layer,
 * while next layer can be fully connected or output layer.
 * This layer calculates weighted sum of outputs from the previous layers, and applies activation function to that sum.
 *
 * @see ActivationType
 * @see ActivationFunction
 * @author Zoran Sevarac
 */
public final class FullyConnectedLayer extends AbstractLayer {

    private static Logger LOG = Logger.getLogger(DeepNetts.class.getName());

     /**
     * Creates an instance of fully connected layer with specified width (number
     * of neurons) and sigmoid activation function.
     *
     * @param width layer width / number of neurons in this layer
     */
    public FullyConnectedLayer(int width) {
        this.width = width;
        this.height = 1;
        this.depth = 1;

        setActivationType(ActivationType.SIGMOID);
    }

    /**
     * Creates an instance of fully connected layer with specified width (number
     * of neurons) and activation function type.
     *
     * @param width layer width / number of neurons in this layer
     * @param actType activation function type to use in this layer
     * @see ActivationFunctions
     */
    public FullyConnectedLayer(int width, ActivationType actType) {
        this(width);
        setActivationType(actType);
    }

    /**
     * Creates all internal data structures: inputs, weights, biases, outputs, deltas,
     * deltaWeights, deltaBiases prevDeltaWeights, prevDeltaBiases. Init weights
     * and biases. This method is called from network builder during initialization
     */
    @Override
    public void init() {
        // check prev and next layers and throw exception if its illegall architecture
        if (!(prevLayer instanceof InputLayer ||
            prevLayer instanceof FullyConnectedLayer ||
            prevLayer instanceof MaxPoolingLayer ||
            prevLayer instanceof ConvolutionalLayer)) throw new DeepNettsException("Bad network architecture! Fully Connected Layer can be connected only to Input, Maxpooling or Convolutional layer as previous layer.");

        if (!(nextLayer instanceof FullyConnectedLayer ||
            nextLayer instanceof OutputLayer)) throw new DeepNettsException("Bad network architecture! Fully Connected Layer can only be connected only to Fully Connected Layer or Output layer as next layer");

        inputs = prevLayer.outputs;
        outputs = new Tensor(width);
        deltas = new Tensor(width);

        if (prevLayer instanceof FullyConnectedLayer) {
            weights = new Tensor(prevLayer.width, width);
            deltaWeights = new Tensor(prevLayer.width, width);
            gradients = new Tensor(prevLayer.width, width);

            WeightsInit.xavier(weights.getValues(), prevLayer.width, width);
            // WeightsInit.randomize(weights.getValues());

        } else if ((prevLayer instanceof MaxPoolingLayer) || (prevLayer instanceof ConvolutionalLayer) || (prevLayer instanceof InputLayer)) {
            weights = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);
            deltaWeights = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);
            gradients = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);

            int totalInputs = prevLayer.getWidth() * prevLayer.getHeight() * prevLayer.getDepth();
            WeightsInit.xavier(weights.getValues(), totalInputs, width);
        }

        biases = new float[width];
        deltaBiases = new float[width];
        WeightsInit.randomize(biases);
    }

    @Override
    public void forward() {
        if (prevLayer instanceof FullyConnectedLayer) {
            outputs.copyFrom(biases);                                                       // first use (add) biases to all outputs
            // put gtters cols in fibal locals
            for (int outCol = 0; outCol < outputs.getCols(); outCol++) {                    // for all outputs in this layer
                for (int inCol = 0; inCol < inputs.getCols(); inCol++) {                    // iterate all inputs from prev layer
                    outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));    // and add weighted sum to outputs
                }

                //outputs.set(outCol, activation.getValue(outputs.get(outCol)));
            }
            outputs.apply(activation::getValue);
        } // if previous layer is MaxPooling, Convolutional or input layer (2D or 3D) - TODO: posto je povezanost svi sa svima ovo mozda moze i kao 1d na 1d niz, verovatno je efikasnije
        else if ((prevLayer instanceof MaxPoolingLayer) || (prevLayer instanceof ConvolutionalLayer) || (prevLayer instanceof InputLayer)) {
            outputs.copyFrom(biases);                                             // first use (add) biases to all outputs
            for (int outCol = 0; outCol < outputs.getCols(); outCol++) {          // for all neurons/outputs in this layer
                for (int inDepth = 0; inDepth < inputs.getDepth(); inDepth++) {   // iterate depth from prev/input layer
                    for (int inRow = 0; inRow < inputs.getRows(); inRow++) {      // iterate current channel by height (rows)
                        for (int inCol = 0; inCol < inputs.getCols(); inCol++) {   // iterate current feature map by width (cols)
                            outputs.add(outCol, inputs.get(inRow, inCol, inDepth) * weights.get(inCol, inRow, inDepth, outCol)); // add to weighted sum of all inputs (TODO: ako je svaki sa svima to bi mozda moglo da bude i jednostavno i da se prodje u jednom loopu a ugnjezdeni loopovi bi se lakse paralelizovali)
                        }
                    }
                }
                // apply activation function to all weigthed sums stored in outputs
                outputs.set(outCol, activation.getValue(outputs.get(outCol)));
            }
        }
    }

    @Override
    public void backward() {
        if (!batchMode) { // if online mode reset deltaWeights and deltaBiases to zeros
            deltaWeights.fill(0);
            Arrays.fill(deltaBiases, 0);
        }

        deltas.fill(0); // reset current delta

        // STEP 1. propagate weighted deltas from next layer (which can be output or fully connected) and calculate deltas for this layer
        for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) {   // for every neuron/delta in this layer
            for (int ndCol = 0; ndCol < nextLayer.deltas.getCols(); ndCol++) { // iterate all deltas from next layer
                deltas.add(deltaCol, nextLayer.deltas.get(ndCol) * nextLayer.weights.get(deltaCol, ndCol)); // calculate weighted sum of deltas from the next layer
            }

            final float delta = deltas.get(deltaCol) * activation.getPrime(outputs.get(deltaCol));
            deltas.set(deltaCol, delta);
        } // end sum weighted deltas from next layer

        // STEP 2. calculate delta weights if previous layer is Dense (2D weights matrix) - optimize
        if ((prevLayer instanceof FullyConnectedLayer)) {
//            Optimizer opt = new SGDOptimizer(); // create instance in init method
//            opt.optimize(this);
            for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) { // this iterates neurons (weights depth)
                for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                    final float grad = deltas.get(deltaCol) * inputs.get(inCol); // gradient dE/dw
                    gradients.set(inCol, deltaCol, grad);

                    float deltaWeight = 0;

                    switch (optimizer) {
                        case SGD:
                            deltaWeight = Optimizers.sgd(learningRate, grad);
                            break;
                    }

                    deltaWeights.add(inCol, deltaCol, deltaWeight);

                }

                float deltaBias = 0;
                switch (optimizer) {
                    case SGD:
                        deltaBias = Optimizers.sgd(learningRate, deltas.get(deltaCol));
                        break;
                }

                deltaBiases[deltaCol] += deltaBias;
            }
        } else if ((prevLayer instanceof InputLayer)
                || (prevLayer instanceof ConvolutionalLayer)
                || (prevLayer instanceof MaxPoolingLayer)) {

            for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) {
                for (int inDepth = 0; inDepth < inputs.getDepth(); inDepth++) {
                    for (int inRow = 0; inRow < inputs.getRows(); inRow++) {
                        for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                            final float grad = deltas.get(deltaCol) * inputs.get(inRow, inCol, inDepth);
                            gradients.set(inCol, inRow, inDepth, grad);

                            float deltaWeight = 0;
                            switch (optimizer) {
                                case SGD:
                                    deltaWeight = Optimizers.sgd(learningRate, grad);
                                    break;
                            }

                            deltaWeights.add(inCol, inRow, inDepth, deltaCol, deltaWeight);
                        }
                    }
                }

                float deltaBias = 0;
                switch (optimizer) {
                    case SGD:
                        deltaBias = Optimizers.sgd(learningRate, deltas.get(deltaCol));
                        break;
                }

                deltaBiases[deltaCol] += deltaBias;
            }
        }
    }

    @Override
    public void applyWeightChanges() {
        if (batchMode) {
            deltaWeights.div(batchSize);
            Tensor.div(deltaBiases, batchSize);
        }

        weights.add(deltaWeights);
        Tensor.add(biases, deltaBiases);

        if (batchMode) {
            deltaWeights.fill(0);
            Tensor.fill(deltaBiases, 0);
        }

    }

}
