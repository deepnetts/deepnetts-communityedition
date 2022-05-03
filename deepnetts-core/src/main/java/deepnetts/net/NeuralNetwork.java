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

package deepnetts.net;

import deepnetts.eval.Evaluators;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.InputLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.loss.BinaryCrossEntropyLoss;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.net.loss.LossFunction;
import deepnetts.net.loss.LossType;
import deepnetts.net.loss.MeanSquaredErrorLoss;
import deepnetts.net.train.Trainer;
import deepnetts.net.train.TrainerProvider;
import deepnetts.util.Tensor;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.data.MLDataItem;
import java.io.IOException;
import java.io.ObjectInputStream;

/**
 * Base class for all neural networks in DeepNetts.
 * Holds a list of abstract layers and loss function.
 * Provides methods for forward and backward calculation, and to access input and output layers.
 * Also provides network and output labels.
 *
 * @see AbstractLayer
 * @see LossFunction
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class NeuralNetwork<T extends Trainer> implements TrainerProvider<T>, Serializable {

    private static final long serialVersionUID = 1L;

    private T trainer;

    /**
     * Collection of all layers in this network (including input(first), output(last) and hidden(in between)).
     * As a minimum neural network must have an input and output layer.
     */
    private final List<AbstractLayer> layers;

    /**
     * Loss function
     * Loss function represents total network error for some data, and network learns by minimizing that error.
     * Commonly used types of loss functions are Mean Squared Error for regression problems and and Cross Entropy for classification problems.
     */
    private LossFunction lossFunction;

    /**
     * Input layer.
     * This layer accepts external inputs and sends them to the next layer
     */
    private InputLayer inputLayer;

    /**
     * Output layer.
     * This layer is the final step of processing network's input and its output is network's output.
     */
    private OutputLayer outputLayer;

    /**
     * Labels for network outputs (classes)
     */
    private String[] outputLabels;


    private Tensor inputWrapper;

    /**
     * Network's label
     */
    private String label;
    private float regularizationSum=0;


    protected NeuralNetwork() {
       layers = new ArrayList();
    }


    /**
     * Sets network input vector and triggers forward pass.
     *
     * @param inputs  input tensor
     */
    public void setInput(Tensor inputs) {
        inputLayer.setInput(inputs);
        forward();
    }

//    public void setInput(float[] inputs) {
////        inputWrapper.setValues(inputs); // also set size / diemnsions / shape of this vector - da li ova metoda da bude ovde??? mozda samo ff ne i zconv!
//        setInput(inputWrapper);
//    }

    /**
     * Returns network's output.
     *
     * @return network's output
     */
    public float[] getOutput() {
        return outputLayer.getOutputs().getValues();
    }

    public void setOutputError(float[] outputErrors) {
        outputLayer.setOutputErrors(outputErrors);
    }

    public void train(DataSet<? extends MLDataItem> trainingSet) {
        trainer.train(trainingSet);
    }

    public EvaluationMetrics test(DataSet<MLDataItem> testSet) {
        if (getLossFunction() instanceof CrossEntropyLoss ||
            getLossFunction() instanceof BinaryCrossEntropyLoss) {
          return Evaluators.evaluateClassifier(this, testSet);
        }

        return Evaluators.evaluateRegressor(this, testSet);
    }

    /**
     * Apply calculated weight changes to all layers.
     */
    public void applyWeightChanges() {
        layers.forEach((layer) -> layer.applyWeightChanges()); // this can be parellelized since all layers are allraedy calculated - each layer cann apply changes in its own thread
    }

    public void forward() {
        for (int i = 1; i < layers.size(); i++) {   // starts from 1 to skip input layer
            layers.get(i).forward();
        }
    }

    public void backward() {
        // perfrom backward pass on all layers starting from last
        for (int i = layers.size() - 1; i > 0; i--) {
            layers.get(i).backward();
        }
    }

    protected void addLayer(AbstractLayer layer) {
        layers.add(layer);
    }

    public List<AbstractLayer> getLayers() {
        return layers;
    }

    public InputLayer getInputLayer() {
       return inputLayer;
    }

    public OutputLayer getOutputLayer() {
        return outputLayer;
    }
  
    public void setOutputLabels(String... outputLabels) {
        this.outputLabels = outputLabels;
    }    

    public String[] getOutputLabels() {
        return outputLabels;
    }

    public String getOutputLabel(int i) {
        return outputLabels[i];
    }

    protected void setInputLayer(InputLayer inputLayer) {
        this.inputLayer = inputLayer;
    }

    protected void setOutputLayer(OutputLayer outputLayer) {
        this.outputLayer = outputLayer;
    }

    public LossFunction getLossFunction() {
        return lossFunction;
    }

    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
        if (lossFunction instanceof MeanSquaredErrorLoss) {
            outputLayer.setLossType(LossType.MEAN_SQUARED_ERROR);
        } else if ((lossFunction instanceof CrossEntropyLoss) || (lossFunction instanceof BinaryCrossEntropyLoss)) {
            outputLayer.setLossType(LossType.CROSS_ENTROPY);
        }
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public float getL2Reg() {
        regularizationSum=0;
        for (int i = 1; i < layers.size(); i++) {   // starts from 1 to skip input layer
            regularizationSum += layers.get(i).getL2();
        }
        return regularizationSum;
    }

    public float getL1Reg() {
        regularizationSum=0;
        for (int i = 1; i < layers.size(); i++) {   // starts from 1 to skip input layer
            regularizationSum += layers.get(i).getL1();
        }
        return regularizationSum;
    }

    @Override
    public T getTrainer() {
        return trainer;
    }

    @Override
    public void setTrainer(T trainer) {
        this.trainer = trainer;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        layers.stream().forEach( layer -> sb.append(layer.toString()).append(System.lineSeparator()) );
                
        return sb.toString();
    }
    
}