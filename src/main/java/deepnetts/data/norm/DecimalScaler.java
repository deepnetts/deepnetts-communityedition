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

import deepnetts.tensor.Tensors;
import java.io.Serializable;
import javax.visrec.ml.data.DataSet;
import deepnetts.tensor.TensorBase;
import deepnetts.data.MLDataItem;

/**
 * Decimal scale normalization for the given data set.
 * Divides all inputs and outputs with 10^k so they are all scaled to [0,1]
 * 
 * @author Zoran Sevarac
 */
public class DecimalScaler extends AbstractScaler {
    private float[] inputDivisor;
    private float[] outputDivisor;
                
    /**
     * Creates a new instance of max normalizer initialized to max values in given data set.
     * 
     * @param dataSet 
     */
    public DecimalScaler(DataSet<MLDataItem> dataSet) {
        // find max values for each component of input and output tensor/vector
        inputDivisor = dataSet.get(0).getInput().copy().getValues();
        outputDivisor = dataSet.get(0).getTargetOutput().copy().getValues();
        
        // nadji maksimume za sve kolone
        for(MLDataItem item : dataSet) {
            inputDivisor = Tensors.absMax(item.getInput().getValues(), inputDivisor);
            outputDivisor = Tensors.absMax(item.getTargetOutput().getValues(), outputDivisor);
        }   
        
        // onda za svaki vektor nadji decimalnu skalu, 1, 10, 100, 1000 while (x>1) { x = x / 10.0f; scale++;}        
        for(MLDataItem item : dataSet) { 
            // i za svaku komponentu ulaznog ili izlaznog vektora
            inputDivisor = getDecimalScaleFor(item.getInput().getValues());
            outputDivisor = getDecimalScaleFor(item.getTargetOutput().getValues());        
        }
                
    }
    
      private float[] getDecimalScaleFor(float[] values) {
          float[] decimalScales = new float[values.length];
          
          for(int i=0; i<values.length; i++) {
              decimalScales[i] = getDecimalScaleFor(values[i]);
          }
          
          return decimalScales;
      }
    
    private float getDecimalScaleFor(float val) {
        float scale = 1;
        val = Math.abs(val);
        
         while (val>1) {
             scale = scale * 10;
             val = val / scale; 
         }         
        return scale; 
    }
        
    /**
     * Performs normalization on the given inputs.
     * 
     * @param dataSet data set to normalize
     */
    @Override
    public void apply(DataSet<MLDataItem> dataSet) {
        for(MLDataItem item : dataSet) {
            item.getInput().div(inputDivisor);
            item.getTargetOutput().div(outputDivisor);
        }    
    }
    
    public void scaleInput(TensorBase input) {
        input.div(inputDivisor);
    }    
       
}
