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

package deepnetts.examples;

import deepnetts.data.DataSets;
import deepnetts.eval.Evaluators;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.Tensor;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;

/**
 * Minimal example for linear regression using FeedForwardNetwork.
 * Fits a straight line (linear function y=k*x+n) through the data.
 * Uses a single layer with one output and linear activation function, and Mean Squared Error for Loss function.
 * Linear regression can be used to roughly estimate a general trend in data.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class BostonHouses {

    public static void main(String[] args) throws IOException {

            int inputsNum = 1;
            int outputsNum = 1;
            String csvFilename = "datasets/bostonsredjen-2kolone.csv";

            // load and create data set from csv file
            DataSet dataSet = DataSets.readCsv(csvFilename , inputsNum, outputsNum, true);
            DataSet[] trainAndTestSet = dataSet.split(0.6);

            // create neural network using network specific builder
            FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                    .addInputLayer(inputsNum)
                    .addFullyConnectedLayer(3, ActivationType.TANH)
                    .addOutputLayer(outputsNum, ActivationType.LINEAR)
                    .lossFunction(LossType.MEAN_SQUARED_ERROR)
                    .build();
            
            neuralNet.getTrainer().setMaxError(0.006f);

            neuralNet.train(trainAndTestSet[0]);

            EvaluationMetrics pm = Evaluators.evaluateRegressor(neuralNet, trainAndTestSet[1]);
            System.out.println(pm);

            // perform prediction for some input value
            neuralNet.setInput(Tensor.create(1, 1, new float[] {0.2f}));
            System.out.println("Predicted price of the house is for 8 :" + neuralNet.getOutput()[0]);//*50);
    }

}
