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

/**
 * Iris Flowers Classification Problem.
 * Hello world classification example: classify flowers into one of 3 possible categories, 
 * based on 4 input features which represent flower several flower dimensions.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class IrisFlowersClassifier {

    public static void main(String[] args) throws DeepNettsException, IOException {

        int numInputs = 4;  // corresponds to number of input features
        int numOutputs = 3; // corresponds to number of possible classes/categories
        
        // load iris data  set from csv file
        DataSet dataSet = DataSets.readCsv("datasets/iris_data_normalised.txt", numInputs, numOutputs, true);
        // split loaded data into 60 : 40% ratio
        DataSet[] trainTestSet = dataSet.split(0.6, 0.4);

        // create instance of multi addLayer percetpron using builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addFullyConnectedLayer(5, ActivationType.TANH)
                .addOutputLayer(numOutputs, ActivationType.SOFTMAX)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(456)
                .build();

        // create and configure instanceof backpropagation trainer
//        BackpropagationTrainer trainer = neuralNet.getTrainer();
//        trainer.setMaxError(0.04f);
//        trainer.setLearningRate(0.01f);
//        trainer.setMomentum(0.9f);
//        trainer.setOptimizer(OptimizerType.MOMENTUM);

        neuralNet.train(trainTestSet[0]);
         
//        // evaluate/test classifier
//        ClassifierEvaluator evaluator = new ClassifierEvaluator();
//        EvaluationMetrics em = evaluator.evaluate(neuralNet, trainTestSet[1]);
//        System.out.println("CLASSIFIER EVALUATION METRICS");
//        System.out.println(em);
//        System.out.println("CONFUSION MATRIX");
//        ConfusionMatrix cm = evaluator.getConfusionMatrix();
//        System.out.println(cm);
    }
}
