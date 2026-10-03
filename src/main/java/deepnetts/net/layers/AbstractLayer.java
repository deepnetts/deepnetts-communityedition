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

import deepnetts.accl.AcceleratorHandle;
import deepnetts.core.DeepNetts;
import deepnetts.net.Mode;
import deepnetts.net.NetworkType;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.tensor.TensorBase;
import java.io.Serializable;
import deepnetts.net.layers.activation.ActivationFunction;
import deepnetts.net.train.opt.Optimizer;
import deepnetts.net.weights.RandomWeightsType;
import deepnetts.tensor.Tensor1D;
import deepnetts.util.DeepNettsThreadPool;

/**
 * Base class for different types of layers. Provides common functionality for
 * all type of layers: layer dimensions, inputs, outputs, connection to previous
 * and/or next layer, activation function and abstract methods for
 * initialization, forward and backward pass.
 *
 * @param <O> output tensor class
 * @param <W> weights tensor class
 */
public abstract class AbstractLayer<I extends TensorBase, O extends TensorBase, W extends TensorBase> implements Layer<O>, Serializable {

    private static final long serialVersionUID = -3972836675081087082L;

    /**
     * Previous layer in network.
     */
    protected AbstractLayer prevLayer;

    /**
     * Next layer in network.
     */
    protected AbstractLayer nextLayer;

    protected NetworkType networkType;

    /**
     * Input weight matrix / connectivity matrix for previous layer. Used in
     * FullyConnected and OutputLayer. MaxPooling does not have Weights and
     * ConvolutionalLayer has weights in filters.
     */
    protected W weights;

    /**
     * Inputs to this layer. A reference to outputs in previous layer, or
     * external input in input layer).
     */
    protected I inputs;

    /**
     * Layer outputs.
     */
    protected O outputs;

    /**
     * Deltas used for learning.
     */
    protected O deltas; //

    /**
     * Weight changes for current and previous iteration.
     */
    protected W deltaWeights, prevDeltaWeights;

    /**
     * Gradients of a loss function calculates during a backward pass.
     */
    protected W gradients; // ovo moze biti transient polje ali mora da se reinicijalizuje prilikom deserijalziacije

    /**
     * Activation function for this layer.
     */
    protected ActivationFunction activation;

    /**
     * Type of activation function for this layer.
     */
    protected ActivationType activationType;

    /**
     * Learning rate for this layer.
     */
    protected float learningRate = 0.1f;

    protected float momentum = 0f;

    protected float regL2 = 0, regL1 = 0;

    protected OptimizerType optimizerType = OptimizerType.SGD;

    protected boolean batchMode = false;
    protected int batchSize = 0;

    protected int width, height, depth; // layer dimensions - width and height

    // biases are used by output, fully connected and convolutional layers   
    protected Tensor1D biases;
    protected Tensor1D deltaBiases;
    protected Tensor1D prevDeltaBiases;

    protected boolean trainable = true;

    protected Optimizer optimizer;

    protected RandomWeightsType randomWeightsType = RandomWeightsType.XAVIER;

    protected transient AcceleratorHandle cudaHandles;
    protected transient Forward forwardImpl;
    protected transient Backward backwardImpl;
    protected transient DeepNettsThreadPool threadPool;
    protected transient int numThreads;

    protected transient Mode mode;

    public AbstractLayer(ActivationType activationType) {
        this.activationType = activationType;
        this.activation = ActivationFunction.create(activationType);
    }

    /**
     * This method should implement layer initialization in subclasses, when a
     * layer is added to the network (create weights, outputs, deltas,
     * randomization etc.).
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

    public NetworkType getNetworkType() {
        return networkType;
    }

    public void setNetworkType(NetworkType networkType) {
        this.networkType = networkType;
    }

    public W getWeights() {
        return weights;
    }

    public Tensor1D getBiases() {
        return biases;
    }

    public void setBiases(Tensor1D biases) {
        this.biases = biases;
    }

    @Override
    public final O getOutputs() {
        return outputs;
    }

    @Override
    public final O getDeltas() {
        return deltas;
    }

    public final W getGradients() {
        return gradients;
    }

    public W getDeltaWeights() {
        return deltaWeights;
    }

    public W getPrevDeltaWeights() {
        return prevDeltaWeights;
    }

    public void setPrevDeltaWeights(W prevDeltaWeights) {
        this.prevDeltaWeights = prevDeltaWeights;
    }

    public Tensor1D getPrevDeltaBiases() {
        return prevDeltaBiases;
    }

    public Tensor1D getDeltaBiases() {
        return deltaBiases;
    }

    public final void setOutputs(O outputs) {
        this.outputs = outputs;
    }

    public void setWeights(W weights) {
        this.weights = weights;
    }

    public void setWeights(String weightStr) {
        weights.setValuesFromString(weightStr);
    }

    public final void setDeltas(O deltas) {
        this.deltas = deltas;
    }

    public ActivationFunction getActivation() {
        return activation;
    }

//    public void setActivation(ActivationFunction activation) {
//        this.activation = activation;
//    }
    public Optimizer getOptimizer() {
        return optimizer;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public boolean isBatchMode() {
        return batchMode;
    }

    public void setBatchMode(boolean batchMode) {
        this.batchMode = batchMode;
    }

    public int getBatchSize() {
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

    public OptimizerType getOptimizerType() {
        return optimizerType;
    }

    public void setOptimizerType(OptimizerType optType) {
        this.optimizerType = optType;
        optimizer = Optimizer.create(optType, this);
    }

    public ActivationType getActivationType() {
        return activationType;
    }

    public final void setActivationType(ActivationType activationType) {
        this.activationType = activationType;
        if (activationType != ActivationType.SOFTMAX) this.activation = ActivationFunction.create(activationType); // we use different layer for softmax
    }
    
    public float getL1WeightSum() {
        return weights.sumAbs();
    }

    public float getL2WeightSum() {
        return weights.sumSqr();
    }

    public float getL2Regularization() {
        return regL2;
    }

    public void setL2Regularization(float regL2) {
        this.regL2 = regL2;
    }

    public float getL1Regularization() {
        return regL1;
    }

    public void setL1Regularization(float regL1) {
        this.regL1 = regL1;
    }

    public boolean isTrainable() {
        return trainable;
    }

    /**
     * Set trainable to false to freeze learned weights.
     *
     * @param trainable
     */
    public void setTrainable(boolean trainable) {
        this.trainable = trainable;
    }

    public void initTransientFields() {

    }

    public void setCudaHandles(AcceleratorHandle cudaHandles) {
        this.cudaHandles = cudaHandles;
    }

    public Forward getForwardAcc() {
        return forwardImpl;
    }

    public Backward getBackwardAcc() {
        return backwardImpl;
    }
    
    public Mode getMode() {
        return mode;
    }

    public void setMode(Mode mode) {
        this.mode = mode;
    }

    public void setThreadPool(DeepNettsThreadPool threadPool) {
        this.threadPool = threadPool;
    }

    protected int getNumThreads(int numElements) {
        final int maxTreads = DeepNetts.getInstance().getMaxThreads();
        int numThr = 0;
        // ovde zapravo treba vracati max threads
        if (numElements < 256) {
            return 1; // verovatno negd eizmedju ima i 2 - tipa 16 * 16 = 256
        } else if (numElements < 1024) {
            numThr = 2; // verovatno negd eizmedju ima i 2 - tipa 16 * 16 = 256
        } else if (numElements < 4096) {
            numThr = 3;
        } else if (numElements < 9216) {
            numThr = 3;
        } else if (numElements < 16384) {
            numThr = 4;
        } else if (numElements < 65536) {
            numThr = 4;
        } else if (numElements < 262144) {
            numThr = 5;
        } else if (numElements < 1048576) {
            numThr = 10;
        } else if (numElements < 4194304) {
            numThr = 12;
        } //else if (numElements >= 4194304) numThr= 12;
        else {
            numThr = 12;
        }

        if (numThr > maxTreads) {
            numThr = maxTreads;
        }

        return numThr;
    }

}
