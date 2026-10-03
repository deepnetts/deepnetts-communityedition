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

import deepnetts.accl.AcceleratorTensorBridge;
import deepnetts.core.DeepNetts;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;

/**
 * Input layer in a neural network. It is always the first layer in the network.
 * It accepts external input, and sends it for processing to the next layer in a
 * network. Inputs are given as tensors of float values.
 *
 * @see AbstractLayer
 * @author Zoran Sevarac
 */
public class InputLayer extends AbstractLayer {
    // TODO: add batch size as first dim in tensor

    private static final long serialVersionUID = 6519195289402298178L;
    private int tensorDim = 0;

    /**
     * Creates input layer with specified width, height, and depth (number of
     * channels).
     *
     * @param width layer width
     * @param height layer height
     * @param depth layer depth (number of input channels)
     */
    public InputLayer(int depth, int width, int height) {
        super(ActivationType.LINEAR);
        this.width = width;
        this.height = height;
        this.depth = depth;  // number of input channels / depth
        this.tensorDim = 3;
        init();
    }

    public InputLayer(int batchSize, int depth, int width, int height) {
        super(ActivationType.LINEAR);
        this.width = width;
        this.height = height;
        this.depth = depth;  // number of input channels / depth
        this.batchSize = batchSize;
        this.tensorDim = 4;
        init();
    }

    /**
     * Creates input layer with specified width and height, and depth=1 (single
     * depth/channel).
     *
     * @param width layer width
     * @param batchSize layer height or batchSize
     */
    public InputLayer(int width, int batchSize) {
        super(ActivationType.LINEAR);
        this.width = width;
        this.height = 1;
        this.batchSize = batchSize;
        this.depth = 1; // using single input channel
        this.tensorDim = 2;
        init();
    }

    public InputLayer(int width, int batchSize, boolean isGPU) { // height je batch size za single dim layers
        super(ActivationType.LINEAR);
        this.width = width;
        this.height = 1;
        this.batchSize = batchSize;
        this.depth = 1; // using single input channel
        if (!isGPU) {
            this.tensorDim = 2;
        } else {
            this.tensorDim = 4;
        }
        init();
    }

    /**
     * Creates input layer with specified width, and with height and depth
     * equals to one.
     *
     * @param width layer width
     */
    public InputLayer(int width) { // @fix: treba da bude size i height da ide na size
        super(ActivationType.LINEAR);
        this.width = width;
        this.height = 1;
        this.depth = 1;
        this.tensorDim = 1;
        init();
    }

    /**
     * Initialize this layer in network.
     */
    @Override
    public final void init() { // ovaj metod je sad nepotreban kad setInput setuje tensor - potreban je zbog inicijalizacije
        if (tensorDim == 1) {
            inputs = new Tensor1D(width);
        } else if (tensorDim == 2) {
            inputs = new Tensor2D(width, batchSize);
        } else if (tensorDim == 3) {
            inputs = new Tensor3D(depth, height, width);
        } else if (tensorDim == 4) {
            inputs = new Tensor4D(batchSize, depth, width, height);
        }
        outputs = inputs;  // in input layer outputs are pointing to the same tensor as inputs        
    }

    /**
     * Sets network input
     *
     * @param in input matrix/array
     */
    public void setInput(TensorBase in) {
        // TODO: check input tensor dimensions and throw exception if they dont match: 1d, 2d, 3d
//        if (height == 1 && depth == 1) {
//            inputs = new Tensor1D(width);
//        } else {
//            inputs = new Tensor3D(depth, height, width);
//        }

        this.inputs = in;
        this.outputs = inputs;
        // ovde poziv cudaAcc  da kopira tensor on device
        if (DeepNetts.getInstance().useCuda()) {
            AcceleratorTensorBridge cuTb = (AcceleratorTensorBridge) in.getOrCreateAccBridge(); // da li ovde cretae ili getOrCReate
            if (!cuTb.isAllocatedOnDev()) {
                cuTb.allocateAndCopyToDev();
            }
        }
    }

    /**
     * This method does nothing in the input layer, and should never be called.
     */
    @Override
    public void forward() {
        throw new IllegalStateException("This method does nothing in this class and should never be called");
    }

    /**
     * This method does nothing in the input layer, and should never be called.
     */
    @Override
    public void backward() {
        throw new IllegalStateException("This method does nothing in this class and should never be called");
    }

    /**
     * This method does nothing in input layer.
     */
    @Override
    public void applyWeightChanges() {
    }

    @Override
    public String toString() {
        if (height == 1 && depth == 1) {
            return "Input Layer { width:" + width + ", height:" + height + ", depth:" + depth + " }";
        } else {
            return "Input Layer { width:" + width + ", height:" + height + ", depth:" + depth + " }";
        }
    }

    public int getTensorDim() {
        return tensorDim;
    }

}
