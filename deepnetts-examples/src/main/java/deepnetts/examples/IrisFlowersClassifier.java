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
package deepnetts.examples;

import deepnetts.data.DataSets;
import deepnetts.data.preprocessing.scale.MaxScaler;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.eval.ConfusionMatrix;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.data.preprocessing.Scaler;

/**
 * Iris Flowers Classification Problem.
 * Hello world classification example: classify flowers into one of 3 possible categories, 
 * based on 4 input features which represent flower several flower dimensions.
 * For more info about the iris classification problem and data set see https://en.wikipedia.org/wiki/Iris_flower_data_set
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class IrisFlowersClassifier {

    public static void main(String[] args) throws DeepNettsException, IOException {

        int numInputs = 4;  // corresponds to number of input features/attribute in data set
        int numOutputs = 3; // corresponds to number of categories/classes in data set
        
        // load iris data  set from csv file
        DataSet dataSet = DataSets.readCsv("datasets/iris.csv", numInputs, numOutputs, true);
        
        // scale data to range [0,1] in order to make it suitable for neural network processing
        Scaler scaler = new MaxScaler(dataSet);
        scaler.apply(dataSet);
        
        // split loaded data into training and test set 60 : 40% ratio
        DataSet[] trainTestSet = dataSet.split(0.6, 0.4);
        DataSet trainingSet = trainTestSet[0]; // part of data to use for training
        DataSet testSet = trainTestSet[1]; // part of data set to use for testing/evaluation

        // create instance of feed forward neural network (aka multi layer percetpron) using corresponding builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs) // input layer accepts inputs from data set, and it's size must correspond to number of inputs in data set
                .addFullyConnectedLayer(8, ActivationType.RELU) // hidden fully connected layer enables solving more complex problems
                .addOutputLayer(numOutputs, ActivationType.SOFTMAX) // commonly used activation function in output layer for multi class classification
                .lossFunction(LossType.CROSS_ENTROPY) // commonly used loss function for multi class classification problems
                .randomSeed(456)    // fix ramdomization seed in order to be able to repeat the results - can use nay value
                .build();

        // get and configure instanceof training algorithm for neural network - backpropagation trainer
        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setMaxError(0.04f); // training is stopped when thie error valueis reached
        trainer.setLearningRate(0.01f); // controls the learning step, percent of error used to tune internal weights parametars [0, 0.9]
        trainer.setOptimizer(OptimizerType.MOMENTUM); // use accelerated optimization method
        trainer.setMomentum(0.9f); // ammount of acceleration to use 
        
        // run the training
        neuralNet.train(trainingSet);
         
        // evaluate/test classifier - estimate how it will behave with unseen data
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        EvaluationMetrics em = evaluator.evaluate(neuralNet, testSet);
        System.out.println("CLASSIFIER EVALUATION METRICS"); 
        System.out.println(em); // print classifier test results
        System.out.println("CONFUSION MATRIX"); // print details of the confusion matrix
        ConfusionMatrix cm = evaluator.getConfusionMatrix();
        System.out.println(cm);
    }
}
