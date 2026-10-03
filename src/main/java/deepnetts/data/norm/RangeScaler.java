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

import java.io.Serializable;
import javax.visrec.ml.data.DataSet;
import deepnetts.tensor.TensorBase;
import deepnetts.data.MLDataItem;

/**
 * Normalize data set to specified range.
 * Using formula X = (X-MIN) / (MAX-MIN)
 * Effectively scales all inputs and outputs to specified [MIN,MAX] range
 * Normalizes inputs and outputs.
 * 
 * @author Zoran Sevarac
 */
public class RangeScaler extends AbstractScaler {
    private final float min;
    private final float max;
    private final float range;
    
    /**
     * Creates a new instance of range normalizer initialized to given min and max values.
     * 
     * @param min
     * @param max 
     */
    public RangeScaler(float min, float max) {
        this.min = min;
        this.max = max;
        this.range = max - min;
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
            item.getInput().sub(min);
            item.getInput().div(range);
            item.getTargetOutput().sub(min);
            item.getTargetOutput().div(range);
        }
    }    
    
    public void scaleInput(TensorBase input) {
        input.sub(min);
        input.div(range);
    }
    
}
