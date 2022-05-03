/**  
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation 
 *  based learning and image recognition.
 * 
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */
    
package deepnetts.net.loss;

import deepnetts.net.NeuralNetwork;
import java.io.Serializable;

/**
 * Represents Average Cross Entropy Loss function.
 * This function is used as a loss function for a multi class classification problems.
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class CrossEntropyLoss implements LossFunction, Serializable {
	
	
    private static final long serialVersionUID = 7810738324038602274L;
	
	
    private final float[] outputError;
    private int targetIdx;    
    private float totalError;
    private int patternCount=0;            
    
    public CrossEntropyLoss(NeuralNetwork neuralNet) {
     //   outputLayer = (OutputLayer)neuralNet.getOutputLayer();
        outputError = new float[neuralNet.getOutputLayer().getWidth()];
    }
       
    
    /**
     * Calculates and returns error vector for specified actual and target outputs.
     * 
     * @param actualOutput actual output from the neural network
     * @param targetOutput target/desired output of the neural network
     * @return error vector for specified actual and target outputs
     */
    @Override
    public float[] addPatternError(float[] actualOutput,  float[] targetOutput) {        
        patternCount++;        
        
        for (int i = 0; i < actualOutput.length; i++) {                                     
            outputError[i] = actualOutput[i] - targetOutput[i]; // ovo je dL/dy izvod loss funkcije u odnosu na izlaz ovog neurona - ovo se koristi za deltu izlaznog neurona
            if (targetOutput[i] == 1) {                        
                targetIdx = i; // TODO: this could be set explicitly in data set in order to avoid this if     
            }
        }     

        totalError += (float)Math.log(actualOutput[targetIdx]);        
        
        return outputError;        
    }
    
    @Override
    public void addRegularizationSum(final float reg) {
        totalError += reg;
    }       
    
    @Override
    public float getTotal() {
        return  -totalError / patternCount;
    }
    
    @Override
    public void reset() {
        totalError = 0;
        patternCount=0;
    }

}
