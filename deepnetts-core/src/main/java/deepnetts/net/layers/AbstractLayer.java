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

import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.Tensor;
import java.io.Serializable;
import deepnetts.net.layers.activation.ActivationFunction;

/**
 * Base class for different types of layers (except data/input layer) Provides
 * common functionality for all type of layers
 *
 * Convolutional & Pooling?
 *
 * @author Zoran Sevarac
 */
public abstract class AbstractLayer implements Layer, Serializable {

    private static final long serialVersionUid = 1L;

    /**
     * Previous layer in network
     */
    protected AbstractLayer prevLayer; // pokazuje na prethodnu matricu - i nje uzima ulaz za sebe

    /**
     * Next layer in network
     */
    protected AbstractLayer nextLayer;

    /**
     * Input weight matrix / connectivity matrix for previous layer Koristi se
     * samo u FullyConnected i OutputLayer-u MaxPooling nema Weights
     * ConvolutionalLayer ima weights u filterima
     *
     */
    protected Tensor weights;  // not used by convolutional and pooling layers, maybe it shoul dbe removed from here? only for  fully connected and output layers layers

    /**
     * Inputs to this layer (a reference to outputs matrix in prev layer, or
     * external input in input layer)
     */
    protected Tensor inputs;

    /**
     * Layer outputs
     */
    protected Tensor outputs;

    /**
     * Deltas used for learning
     */
    protected Tensor deltas;

    /**
     * Previous delta sums used by AdaGrad and AdaDelta
     */
    protected Tensor prevGradSqrSum, prevBiasSqrSum, prevDeltaWeightSqrSum, prevDeltaBiasSqrSum;

    /**
     * Weight changes for current and previous iteration
     */
    protected Tensor deltaWeights;

    protected Tensor gradients;

    protected ActivationFunction activation;

    /**
     * Learning rate for this layer
     */
    protected float learningRate = 0.1f;

    protected float momentum = 0;

    /**
     * Activation function for this layer Use function reference for activation
     * functions instead? Function activation;
     */
    protected ActivationType activationType;

    protected OptimizerType optimizer = OptimizerType.SGD;

    // TODO: use method reference for activation and optimization function
    protected boolean batchMode = false;
    protected int batchSize = 0;

    protected int width, height, depth; // layer dimensions - width and height

    protected float[] biases;
    protected float[] deltaBiases;


    /**
     * This method should implement layer initialization when layer is added to
     * network (create weights, outputs, deltas, randomization etc.)
     */
    public abstract void init();

    /**
     * This method should implement forward pass in subclasses
     */
    @Override
    public abstract void forward();

    /**
     * This method should implement backward pass in subclasses
     */
    @Override
    public abstract void backward();

    /**
     * Applies weight changes to current weights Must be diferent for
     * convolutional does nothing for MaxPooling Same for FullyConnected and
     * OutputLayer
     *
     */
    public abstract void applyWeightChanges();

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int getDepth() {
        return depth;
    }

    public AbstractLayer getPrevlayer() {
        return prevLayer;
    }

    public void setPrevLayer(AbstractLayer prevLayer) {
        this.prevLayer = prevLayer;
    }

    public void setNextlayer(AbstractLayer nextlayer) {
        this.nextLayer = nextlayer;
    }

    public AbstractLayer getNextLayer() {
        return nextLayer;
    }

    public Tensor getWeights() {
        return weights;
    }

    public float[] getBiases() {
        return biases;
    }

    public void setBiases(float[] biases) {
        this.biases = biases;
    }

    @Override
    public final Tensor getOutputs() {
        return outputs;
    }

    @Override
    public final Tensor getDeltas() {
        return deltas;
    }

    public final Tensor getGradients() {
        return gradients;
    }

    public Tensor getDeltaWeight() {
        return deltaWeights;
    }

    public float[] getDeltaBiases() {
        return deltaBiases;
    }

    public final void setOutputs(Tensor outputs) {
        this.outputs = outputs;
    }

    public void setWeights(Tensor weights) {
        this.weights = weights;
    }

    public void setWeights(String weightStr) {
        weights.setValuesFromString(weightStr);
    }

    public final void setDeltas(Tensor deltas) {
        this.deltas = deltas;
    }

    public ActivationFunction getActivation() {
        return activation;
    }

    public void setActivation(ActivationFunction activation) {
        this.activation = activation;
    }



    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

//    initialisation after deserialization goes here
//    private void readObject(java.io.ObjectInputStream in)
//        throws IOException, ClassNotFoundException {
//        in.defaultReadObject();
//
//    }

    public boolean isBatchMode() {
        return batchMode;
    }

    public void setBatchMode(boolean batchMode) {
        this.batchMode = batchMode;
    }

    public float getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public void setMomentum(float momentum) {
        this.momentum = momentum;
    }

    public float getMomentum() {
        return momentum;
    }

    public OptimizerType getOptimizer() {
        return optimizer;
    }

    public void setOptimizer(OptimizerType optimizer) {
        this.optimizer = optimizer;
    }

    public ActivationType getActivationType() {
        return activationType;
    }

    public final void setActivationType(ActivationType activationType) {
        this.activationType = activationType;
        this.activation = ActivationFunction.create(activationType);
    }

    public float getL1Norm() {
        return weights.sumAbs();
    }

    public float getL2Norm() {
        return weights.sumSqr();
    }

}
