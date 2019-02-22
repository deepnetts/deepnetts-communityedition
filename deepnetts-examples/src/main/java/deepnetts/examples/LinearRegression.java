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

import deepnetts.data.DataSet;
import deepnetts.data.DataSets;
import deepnetts.eval.PerformanceMeasure;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import java.io.IOException;
import java.util.Arrays;

/**
 * Minimal example for linear regression using FeedForwardNetwork, with generated data set.
 * Fits a straight line through the data.
 * Uses a single layer with one output and linear activation function, and Mean Squared Error for Loss function.
 * Use linear regression to roughly estimate a global trend in data.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class LinearRegression {

    public static void main(String[] args) throws IOException {

            int inputsNum = 1;
            int outputsNum = 1;
            String csvFilename = "datasets/linear.csv";

            // load and create data set from csv file
            DataSet dataSet = DataSets.readCsv(csvFilename , inputsNum, outputsNum);

            // create neural network using network specific builder
            FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                    .addInputLayer(inputsNum)
                    .addOutputLayer(outputsNum, ActivationType.LINEAR)
                    .lossFunction(LossType.MEAN_SQUARED_ERROR)
                    .build();

            BackpropagationTrainer trainer = neuralNet.getTrainer();
            trainer.setMaxError(0.002f)
                   .setMaxEpochs(10000)
                   .setLearningRate(0.01f);

            // train network using loaded data set
            neuralNet.train(dataSet);

            //
            PerformanceMeasure pe = neuralNet.test(dataSet);
            System.out.println(pe);

            // perform prediction for some input value
            float[] predictedOutput = neuralNet.predict(new float[] {0.2f});
            System.out.println("Predicted output for 0.2 :" + Arrays.toString(predictedOutput));
    }
}
