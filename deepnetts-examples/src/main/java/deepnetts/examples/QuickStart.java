package deepnetts.examples;

import deepnetts.examples.util.FileIODebug;
import deepnetts.data.DataSets;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.eval.Evaluators;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;

/**
 * Iris Classification Problem. An example of multi class classification using neural network.
 * This example is using Softmax activation in output layer and Cross Entropy Loss function.
 * Overfits the iris data set
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class QuickStart {

    public static void main(String[] args) throws DeepNettsException, IOException, ClassNotFoundException {
        // load data  set from csv file
        DataSet dataSet = DataSets.readCsv("datasets/iris_data_normalised.txt", 4, 3, true);
        dataSet.shuffle();

        // create instance of multi addLayer percetpron using builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(4)
                .addFullyConnectedLayer(10, ActivationType.TANH)
                .addOutputLayer(3, ActivationType.SOFTMAX)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)
                .build();

        // create and configure instanceof backpropagation trainer
        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setMaxError(0.03f);
        trainer.setMaxEpochs(10000);
        trainer.setBatchMode(false);
        trainer.setLearningRate(0.01f);

        // run training
        neuralNet.train(dataSet);
        
        // evaluate trained network
        EvaluationMetrics em = Evaluators.evaluateClassifier(neuralNet, dataSet);
        System.out.println(em);
        
        // todo: explain evaluation results

        // save trained network to file
        FileIO.writeToFile(neuralNet, "myNeuralNet.dnet");
        
        NeuralNetwork loadedNeuralNet = FileIO.createFromFile(new File("myNeuralNet.dnet"));
           
    }
}
