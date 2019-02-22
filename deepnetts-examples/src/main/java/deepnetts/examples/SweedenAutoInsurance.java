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
 * Minimal example for simple linear regression using FeedForwardNetwork.
 * Fits a straight line through the data.
 * Uses a single addLayer with one output and linear activation function, and Mean Squared Error for Loss function.
 * You can use linear regression to roughly estimate a global trend in data.
 *
 * TODO: dont print accuracy for regression problems!
 *
 * predicting the total payment for all claims in thousands of Swedish Kronor, given the total number of claims.
 *
 * @author Zoran Sevarac
 */
public class SweedenAutoInsurance {

    public static void main(String[] args) throws IOException {

        String datasetFile = "datasets/SweedenAutoInsurance.csv";
        int inputsNum = 1;
        int outputsNum = 1;

        DataSet dataSet = DataSets.readCsv(datasetFile, inputsNum, outputsNum);
        DataSet[] trainTestSet = dataSet.split(0.7);

        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(1)
                .addOutputLayer(1, ActivationType.LINEAR)
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .build();

        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setMaxError(0.01f)
               .setMaxEpochs(100)
               .setLearningRate(0.1f);

        neuralNet.train(trainTestSet[0]);

       // Performance evaluation for the given test set
       PerformanceMeasure pe = neuralNet.test(trainTestSet[1]);
       System.out.println(pe);

       // use model for prediction
       float[] predictedOutput = neuralNet.predict(new float[]{0.18548387f});
       System.out.println(Arrays.toString(predictedOutput));

    }

}
