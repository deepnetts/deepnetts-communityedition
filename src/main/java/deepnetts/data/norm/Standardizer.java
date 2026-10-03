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
import javax.visrec.ml.data.DataSet;
import deepnetts.data.MLDataItem;

/**
 * Performs standardization on inputs in order to get desired statistical properties of the data set (zero mean and one standard deviation).
 *  
 * X = (X - MEAN) / STD
 * 
 * @author Zoran Sevarac
 */
public class Standardizer extends AbstractScaler {
    // only appy to decimal inputs, not binary values
    private final TensorBase mean;
    private final TensorBase std;
    
    
    public Standardizer(DataSet<MLDataItem> dataSet) {
        TensorBase t = dataSet.get(0).getInput();
        mean = (TensorBase)t.clone(); // new TensorBase(t.size()); 
        mean.fill(0);
        std = (TensorBase)t.clone(); //new TensorBase(t.size());
        std.fill(0);
        
        for(MLDataItem item : dataSet) {
            mean.add(item.getInput());
        }         
        mean.div(dataSet.size());
                
        //std = sqrt(sum((x-m)^2)/(n-1));
        
        for(MLDataItem item : dataSet) {
            TensorBase diff = item.getInput().copy();
            diff.sub(mean);
            diff.multiplyElementWise(diff);  // ^2
            std.add(diff);            
        }        
        std.div(dataSet.size()-1);
        std.sqrt();           
    }
        
    @Override
    public void apply(DataSet<MLDataItem> dataSet) {
        for (MLDataItem item : dataSet) {
            item.getInput().sub(mean);
            item.getInput().div(std);
        }
    }
    
    @Override
    public void scaleInput(TensorBase input) {
        input.sub(mean);
        input.div(std);
    }    
    
}
