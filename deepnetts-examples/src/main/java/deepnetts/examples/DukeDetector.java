/**
 * DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 * based learning and image recognition.
 * <p>
 * Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 * <p>
 * This file is part of DeepNetts.
 * <p>
 * DeepNetts is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */

package deepnetts.examples;

import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.eval.ConfusionMatrix;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.FileIO;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import javax.visrec.ml.eval.EvaluationMetrics;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * Convolutional Neural Network that learns to detect Duke images.
 * Example how to create and train convolutional network for image classification.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class DukeDetector {

    static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());

    public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
        int imageWidth = 64;
        int imageHeight = 64;

        String trainingFile = "datasets/DukeSet/train.txt";
        String labelsFile = "datasets/DukeSet/labels.txt";

        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);

        LOGGER.info("Loading images...");

        imageSet.loadLabels(new File(labelsFile));
        imageSet.loadImages(new File(trainingFile));
        //imageSet.zeroMean();

        imageSet.setInvertImages(true);
        imageSet.shuffle();

        // ImageSet[] trainAndTestSet = imageSet.split(0.7, 0.3);

        LOGGER.info("Creating neural network...");

        ConvolutionalNetwork convNet = ConvolutionalNetwork.builder()
                .addInputLayer(imageWidth, imageHeight, 3)
                .addConvolutionalLayer(3, 3, 3, ActivationType.TANH)
                .addMaxPoolingLayer(2, 2, 2)
                .addFullyConnectedLayer(10, ActivationType.TANH)
                .addOutputLayer(1, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();


        convNet.setOutputLabels(imageSet.getTargetColumnsNames());

        LOGGER.info("Training neural network");

        // create a set of convolutional networks and do training, crossvalidation and performance evaluation
        BackpropagationTrainer trainer = new BackpropagationTrainer(convNet);
        trainer.setMaxError(0.05f)
                .setLearningRate(0.01f);
        trainer.train(imageSet);

        // to save neural network to file on disk
        FileIO.writeToFile(convNet, "DukeDetector.dnet");

        // to load neural network from file use FileIO.createFromFile
        // dukeNet = FileIO.createFromFile("DukeDetector.dnet");
        // to serialize network in json use FileIO.toJson
        // System.out.println(FileIO.toJson(dukeNet));

        // to evaluate recognizer with image set
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        EvaluationMetrics evalResults = evaluator.evaluate(convNet, imageSet);
        System.out.println(evalResults);

        ConfusionMatrix cm = evaluator.getConfusionMatrix();
        System.out.println(cm);
        
        // load saved network
        NeuralNetwork loadedNeuralNet = FileIO.createFromFile(new File("DukeDetector.dnet"));
        

        // to use recognizer for single image
//        BufferedImage image = ImageIO.read(new File("/home/zoran/datasets/DukeSet/duke/duke7.jpg"));
//        DeepNettsImageClassifier imageClassifier = new DeepNettsImageClassifier(convNet);
//        ClassificationResults<ClassificationResult> results = imageClassifier.classify(image);

        //   System.out.println(results.toString());
    }

}
