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
import java.io.Serializable;
import javax.visrec.ml.data.DataSet;
import deepnetts.data.MLDataItem;

/**
 * Performs Min Max normalization on the given data set.
 * 
 * @author Zoran Sevarac
 */
public class MinMaxScaler extends AbstractScaler {
    private TensorBase minInput;
    private TensorBase maxInput;
    private TensorBase minOutput;   // ovo ne bi trebalo na outputs da se primenjuje???
    private TensorBase maxOutput;   
    private final TensorBase inDivider;
    private final TensorBase outDivider;
    
    /**
     * Creates a new instance of max normalizer initialized to max values in given data set.
     * 
     * @param dataSet 
     */
    public MinMaxScaler(DataSet<MLDataItem> dataSet) {
        // find min and max values for each component of input and output tensor/vector
        minInput = dataSet.get(0).getInput().copy();
        maxInput = dataSet.get(0).getInput().copy();
        minOutput = dataSet.get(0).getTargetOutput().copy();
        maxOutput = dataSet.get(0).getTargetOutput().copy();
        
        for(MLDataItem item : dataSet) {
            minInput = Tensors.absMin(item.getInput(), minInput);
            maxInput = Tensors.absMax(item.getInput(), maxInput);
            minOutput = Tensors.absMin(item.getTargetOutput(), minOutput);
            maxOutput = Tensors.absMax(item.getTargetOutput(), maxOutput);
        }      
        
        inDivider = maxInput.copy();
        inDivider.sub(minInput); // max-min for inputs
        outDivider = maxOutput.copy();
        outDivider.sub(minOutput); // max - min for outputs        
    }
        
    /**
     * Performs normalization on the given inputs.
     * x = (x-min) / (max-min)
     * 
     * @param dataSet data set to normalize
     */
    @Override
    public void apply(DataSet<MLDataItem> dataSet) {
        for (MLDataItem item : dataSet) {
            item.getInput().sub(minInput);
            item.getInput().div(inDivider);
            item.getTargetOutput().sub(minOutput);
            item.getTargetOutput().div(outDivider);
        }
    }
    
    @Override
    public void scaleInput(TensorBase input) {
        input.sub(minInput);
        input.div(inDivider);
    }
}
