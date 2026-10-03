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

package deepnetts.data.norm;

import deepnetts.tensor.TensorBase;
import deepnetts.tensor.Tensors;
import javax.visrec.ml.data.DataSet;
import deepnetts.data.MLDataItem;

/**
 * Performs max normalization, rescales data to corresponding max value in each column.
 * Scales all values to interval [0, 1], by dividing columns with their corresponding maximum value.
 * Performs normalization on both inputs and outputs.
 * 
 */
public final class MaxScaler extends AbstractScaler {
    private TensorBase maxInputs;
    private TensorBase maxOutputs;
                
    /**
     * Creates a new instance of max normalizer initialized to max values in given data set.
     * 
     * @param dataSet 
     */
    public MaxScaler(DataSet<MLDataItem> dataSet) {
        // find max values for each component of input and output tensor/vector
        maxInputs = dataSet.get(0).getInput().copy();
        maxOutputs = dataSet.get(0).getTargetOutput().copy();
        
        // find max values for all components of input and output vectors
        for(MLDataItem item : dataSet) { // iterate all rows?
            maxInputs = Tensors.absMax(item.getInput(), maxInputs); // ako su nule pretvori ih u jedinicu Tensors.replace 
            maxInputs.replace(0, 1); // since ve dont want to divide with zero
            maxOutputs = Tensors.absMax(item.getTargetOutput(), maxOutputs);
            maxOutputs.replace(0, 1); // since ve dont want to divide with zero
        }        
    }
        
    /**
     * Performs normalization on the given inputs.
     * 
     * @param dataSet data set to normalize
     */
    @Override
    public void apply(DataSet<MLDataItem> dataSet) {
        // todo: prevent/catch division by zero - if something is zero then everything is zero
        for(MLDataItem item : dataSet) {
            item.getInput().div(maxInputs);
            item.getTargetOutput().div(maxOutputs); 
        }    
    }
    
    @Override
    public void scaleInput(TensorBase input) {
        input.div(maxInputs);
    }    

    public TensorBase getMaxInputs() {
        return maxInputs;
    }

    public void setMaxInputs(TensorBase maxInputs) {
        this.maxInputs = maxInputs;
    }

    public TensorBase getMaxOutputs() {
        return maxOutputs;
    }

    public void setMaxOutputs(TensorBase maxOutputs) {
        this.maxOutputs = maxOutputs;
    }
           
    /**
     * De-normalize given output vector in-place.
     * Multiplies given vector with vector used for normalization, and stores these values in same memory location as input vector.
     * 
     * @param outputs 
     */
    public void deNormalizeOutputs(final TensorBase outputs) {
        outputs.multiplyElementWise(maxOutputs);
    }
    
    public void deNormalizeInputs(final TensorBase inputs) {
        inputs.multiplyElementWise(maxInputs);
    }    
    
    public void normalizeInput(TensorBase input) {
        input.div(maxInputs);
    }    
   
}