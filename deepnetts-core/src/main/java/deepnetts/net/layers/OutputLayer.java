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
import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.Optimizers;
import deepnetts.util.WeightsInit;
import deepnetts.util.Tensor;
import java.util.Arrays;
import deepnetts.net.layers.activation.ActivationFunction;

/**
 * Output layer of a neural network, which gives the final output of a network.
 * It is always the last layer in the network.
 *
 * @author Zoran Sevarac
 */
public class OutputLayer extends AbstractLayer {

    protected float[] outputErrors;
    protected final String[] labels;
    protected LossType lossType;

    /**
     * Creates an instance of output layer with specified width (number of outputs)
     * and sigmoid activation function by default.
     * Outputs are labeled using generic names "Output1, 2, 3..."
     *
     * @param width layer width which represents number of network outputs
     */
    public OutputLayer(int width) {
        this.width = width;
        this.height = 1;
        this.depth = 1;

        labels = new String[depth];
        // generate enumerated class names from 1..n
        for (int i = 0; i < depth; i++) {
            labels[i] = "Output" + i;
        }

        setActivationType(ActivationType.SIGMOID);
        this.activation = ActivationFunction.create(ActivationType.SIGMOID);
    }

    /**
     * Creates an instance of output layer with specified width (number of outputs)
     * and specified activation function.
     * Outputs are labeled using generic names "Output1, 2, 3..."
     *
     * @param width layer width which represents number of network outputs
     * @param actType activation function
     */
    public OutputLayer(int width, ActivationType actType) {
        this.width = width;
        this.height = 1;
        this.depth = 1;

        labels = new String[depth];
        // generate enumerated class names from 1..n
        for (int i = 0; i < depth; i++) {
            labels[i] = "Output" + i;
        }

        setActivationType(actType);
        this.activation = ActivationFunction.create(actType);
    }

    /**
     * Creates an instance of output layer with specified width (number of outputs)
     * which corresponds to number of labels and sigmoid activation function by default.
     * Outputs are labeled with strings specified in labels parameter
     *
     * @param outputLabels labels for network's outputs
     */
    public OutputLayer(String[] outputLabels) {
        this.width = outputLabels.length;
        this.height = 1;
        this.depth = 1;
        this.labels = outputLabels;
        setActivationType(ActivationType.SIGMOID);
        this.activation = ActivationFunction.create(ActivationType.SIGMOID);
    }

    public OutputLayer(String[] labels, ActivationType actType) {
        this(labels);
        setActivationType(actType);

    }

    public final void setOutputErrors(final float[] outputErrors) {
        this.outputErrors = outputErrors;
    }

    public final float[] getOutputErrors() {
        return outputErrors;
    }

    public final LossType getLossType() {
        return lossType;
    }

    public void setLossType(LossType lossType) {
        this.lossType = lossType;
    }

    @Override
    public void init() {
        inputs = prevLayer.outputs;
        outputs = new Tensor(width);
        outputErrors = new float[width];
        deltas = new Tensor(width);

        int prevLayerWidth = prevLayer.getWidth();
        weights = new Tensor(prevLayerWidth, width);
        gradients = new Tensor(prevLayerWidth, width);
        deltaWeights = new Tensor(prevLayerWidth, width);
        WeightsInit.xavier(weights.getValues(), prevLayerWidth, width);

        biases = new float[width];
        deltaBiases = new float[width];
        WeightsInit.randomize(biases);
    }

    /**
     * This method implements forward pass for the output layer.
     *
     * Calculates weighted input and layer outputs using sigmoid function.
     */
    @Override
    public void forward() {
        outputs.copyFrom(biases);
        for (int outCol = 0; outCol < outputs.getCols(); outCol++) {
            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));
            }
           // outputs.set(outCol, ActivationFunctions.calc(activationType, outputs.get(outCol)));
        }
        outputs.apply(activation::getValue);
    }

    /**
     * This method implements backward pass for the output layer.
     */
    @Override
    public void backward() {
        if (!batchMode) {
            deltaWeights.fill(0);
            Arrays.fill(deltaBiases, 0);
        }

        for (int dCol = 0; dCol < deltas.getCols(); dCol++) {

            if (lossType == LossType.MEAN_SQUARED_ERROR) {
                deltas.set(dCol, outputErrors[dCol] * ActivationFunctions.prime(activationType, outputs.get(dCol)));
            } else if (activationType == ActivationType.SIGMOID && lossType == LossType.CROSS_ENTROPY) {
                deltas.set(dCol, outputErrors[dCol]);
            }

            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                final float grad = deltas.get(dCol) * inputs.get(inCol);
                final float deltaWeight = Optimizers.sgd(learningRate, grad);
                gradients.set(inCol, dCol, grad);
                deltaWeights.add(inCol, dCol, deltaWeight); // sum deltaWeight for batch mode
            }

            deltaBiases[dCol] += Optimizers.sgd(learningRate, deltas.get(dCol));
        }
    }

    /**
     * Applies weight changes after one learning iteration or batch
     */
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
