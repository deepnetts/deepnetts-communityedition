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

import deepnetts.util.Tensor;

/**
 * Common base interface for all types of neural network layers.
 * Layer is a basic building block of a neural network, and neural network
 * typically consists of a sequence of layers.
 *
 * @see AbstractLayer
 * @author Zoran Sevarac
 */
public interface Layer {

    /**
     * Performs layer calculation in forward pass of a neural network.
     */
    public void forward();

    /**
     * Performs weight parameters adjustment in backward pass during training of a neural network.
     */
    public void backward();

    /**
     * Returns layer outputs (as a tensor).
     * @return layer output tensor
     */
    public Tensor getOutputs();

    /**
     * Returns layer deltas (as a tensor).
     * Deltas are accumulated errors propagated from the next layer.
     * @return layer delta tensor
     */
    public Tensor getDeltas();

}