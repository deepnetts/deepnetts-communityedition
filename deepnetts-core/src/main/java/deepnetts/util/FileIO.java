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

package deepnetts.util;

import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NetworkType;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.net.loss.MeanSquaredErrorLoss;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.ConvolutionalLayer;
import deepnetts.net.layers.FullyConnectedLayer;
import deepnetts.net.layers.InputLayer;
import deepnetts.net.layers.LayerType;
import deepnetts.net.layers.MaxPoolingLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.layers.SoftmaxOutputLayer;
import deepnetts.net.loss.LossType;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * File utilities for saving and loading neural networks.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class FileIO {

    public static final String NETWORK_FILE_EXT = "dnet";

    /**
     * This class has only static utility methods so we don't need instances
     */
    private FileIO() { }

    /**
     * Serializes specified neural network to file with specified file.
     *
     * @param neuralNet neural network to save
     * @param fileName name of the file
     * @throws IOException if something goes wrong
     */
    public static void writeToFile(NeuralNetwork neuralNet, String fileName) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileName))) {
            oos.writeObject(neuralNet);
        }
    }

    public static void writeToFileAsJson(NeuralNetwork neuralNet, String fileName) throws IOException {
        String jsonStr = toJson(neuralNet);
        try (PrintWriter pw = new PrintWriter(new File(fileName))) {
            pw.print(jsonStr);
        }
    }

    public static <T> T createFromFile(String fileName, Class<T> clazz) throws IOException, ClassNotFoundException {
        T neuralNet;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fileName))) {
            neuralNet = clazz.cast(ois.readObject()) ;
        }
        return neuralNet;
    }

    public static NeuralNetwork createFromFile(File file) throws IOException, ClassNotFoundException {
        NeuralNetwork nnet;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            nnet = (NeuralNetwork) ois.readObject();
        }
        return nnet;
    }


    /**
     * Returns JSON representation of specified neural network object.
     * TODO: add biases
     *
     * @param nnet
     * @return
     */
    public static String  toJson(NeuralNetwork<?> nnet) {
        JSONObject json = new JSONObject();
        JSONArray layers = new JSONArray();

        InputLayer inputLayer= nnet.getInputLayer();

        JSONObject inputLayerJson = new JSONObject();
        inputLayerJson.put("layerType", LayerType.INPUT.toString());
        inputLayerJson.put("width", inputLayer.getWidth());

        if (nnet instanceof ConvolutionalNetwork) { // only conv network has this
            inputLayerJson.put("height", inputLayer.getHeight());
            inputLayerJson.put("channels", inputLayer.getDepth());
         }
        layers.put(inputLayerJson);

        for(AbstractLayer layer : nnet.getLayers()){
            if (layer instanceof ConvolutionalLayer) {
                ConvolutionalLayer convLayer = (ConvolutionalLayer)layer;
                JSONObject convLayerJson = new JSONObject();
                convLayerJson.put("layerType", LayerType.CONVOLUTIONAL);
                convLayerJson.put("filterWidth", convLayer.getFilterWidth());
                convLayerJson.put("filterHeight", convLayer.getFilterHeight());
                convLayerJson.put("channels", convLayer.getDepth()); // channels
                convLayerJson.put("stride", convLayer.getStride());
                convLayerJson.put("activation", convLayer.getActivationType());
                JSONArray filters = new JSONArray();
                for(Tensor filter: convLayer.getFilters()) {
                    filters.put(filter);
                }
                //convLayerJson.put("biases", convLayer.getBiases());

                //convLayerJson.put("filters",filters);

                layers.put(convLayerJson);
            } else if (layer instanceof MaxPoolingLayer) {
                MaxPoolingLayer maxPooling= (MaxPoolingLayer)layer;
                JSONObject poolLayerJson = new JSONObject();
                poolLayerJson.put("layerType", LayerType.MAXPOOLING);
                poolLayerJson.put("filterWidth", maxPooling.getFilterWidth());
                poolLayerJson.put("filterHeight", maxPooling.getFilterHeight());
                poolLayerJson.put("stride", maxPooling.getStride());
                layers.put(poolLayerJson);
            } else if (layer instanceof FullyConnectedLayer) {
                JSONObject fullyConnLayerJson = new JSONObject();
                fullyConnLayerJson.put("layerType", LayerType.FULLY_CONNECTED);
                fullyConnLayerJson.put("width", layer.getWidth());
                fullyConnLayerJson.put("activation", layer.getActivationType());
                //fullyConnLayerJson.put("weights", layer.getWeights());
                //fullyConnLayerJson.put("biases", layer.getBiases());
                layers.put(fullyConnLayerJson);
            } else if (layer instanceof SoftmaxOutputLayer) {
                JSONObject outputLayerJson = new JSONObject();
                outputLayerJson.put("layerType", LayerType.OUTPUT);
                outputLayerJson.put("width", layer.getWidth());
                outputLayerJson.put("activation", layer.getActivationType());
                //outputLayerJson.put("weights", layer.getWeights());
                //outputLayerJson.put("biases", layer.getBiases());
                layers.put(outputLayerJson);
            } else if (layer instanceof OutputLayer) {
                JSONObject outputLayerJson = new JSONObject();
                outputLayerJson.put("layerType", LayerType.OUTPUT);
                outputLayerJson.put("width", layer.getWidth());
                outputLayerJson.put("activation", layer.getActivationType());
                //outputLayerJson.put("weights", layer.getWeights());
                //outputLayerJson.put("biases", layer.getBiases());
                layers.put(outputLayerJson);
            }
        }

        json.put("networkType", NetworkType.Of(nnet.getClass()));
        json.put("layers", layers);
        json.put("lossFunction", LossType.of(nnet.getLossFunction().getClass()));

        return json.toString();
    }

     public static NeuralNetwork createFromJson(String jsonStr) {
        JSONObject obj = new JSONObject(jsonStr);
        return createFromJson(obj);
     }

     public static NeuralNetwork createFromJson(File file) throws FileNotFoundException, IOException {
         BufferedReader br = new BufferedReader(new FileReader(file));
         StringBuilder sb = new StringBuilder();
         String line;
         while((line = br.readLine()) != null) {
             sb.append(line).append(System.lineSeparator());
         }
         return createFromJson(sb.toString());
     }

    public static NeuralNetwork createFromJson(JSONObject jsonObj) {
       String networkType = jsonObj.getString("networkType");

       if (networkType.equals(NetworkType.FEEDFORWARD.toString())) {
           return createFeedForwardNetworkFromJson(jsonObj);
       } else if (networkType.equals(NetworkType.CONVOLUTIONAL.toString())) {
           return createConvolutionalNetworkFromJson(jsonObj);
       }

       throw new RuntimeException("Unknown network type: "+networkType);
    }

    public static ConvolutionalNetwork createConvolutionalNetworkFromJson(JSONObject jsonObj) {
        JSONArray jsonLayers = jsonObj.getJSONArray("layers");

        // switch network type here and use corresponding builder
        ConvolutionalNetwork.Builder builder = new ConvolutionalNetwork.Builder();

        int width, height, channels, filterWidth, filterHeight, stride;
        String activation;

        for(Object jsonLayerObject : jsonLayers) {
            JSONObject layerObj = (JSONObject)jsonLayerObject;

            switch(LayerType.valueOf( layerObj.getString("layerType").toUpperCase() ) ) {
                case INPUT :
                        width = layerObj.getInt("width");
                        height = layerObj.getInt("height");
                        channels = layerObj.getInt("channels");
                        builder.addInputLayer(width, height, channels);
                break;
                case CONVOLUTIONAL :
                        filterWidth = layerObj.getInt("filterWidth");
                        filterHeight = layerObj.getInt("filterHeight");
                        stride = layerObj.getInt("stride");
                        channels = layerObj.getInt("channels");
                        activation = layerObj.getString("activation").toUpperCase();

                        builder.addConvolutionalLayer(filterWidth, filterHeight, channels, stride, ActivationType.valueOf(activation));
                break;
                case MAXPOOLING :
                        filterWidth = layerObj.getInt("filterWidth");
                        filterHeight = layerObj.getInt("filterHeight");
                        stride = layerObj.getInt("stride");
                        builder.addMaxPoolingLayer(filterWidth, filterHeight, stride);
                break;
                case FULLY_CONNECTED :
                        width = layerObj.getInt("width");
                        activation = layerObj.getString("activation").toUpperCase();
                        builder.addFullyConnectedLayer(width, ActivationType.valueOf(activation));
                break;
                case OUTPUT :
                        width = layerObj.getInt("width");
                        activation = layerObj.getString("activation").toUpperCase();

                        if (activation.equals(ActivationType.SIGMOID.toString())) {
                            builder.addOutputLayer(width, OutputLayer.class);
                            builder.lossFunction(MeanSquaredErrorLoss.class);
                        } else if (activation.equals(ActivationType.SOFTMAX.toString())) {
                            builder.addOutputLayer(width, SoftmaxOutputLayer.class);
                            builder.lossFunction(CrossEntropyLoss.class);
                        }
                break;
            }
        }

        // set loss function
        String lossFunction = jsonObj.getString("lossFunction");
        builder.lossFunction(LossType.valueOf(lossFunction));

        ConvolutionalNetwork neuralNet = builder.build();

        return neuralNet;
    }

    public static FeedForwardNetwork createFeedForwardNetworkFromJson(JSONObject jsonObj) {
        JSONArray jsonLayers = jsonObj.getJSONArray("layers");

        FeedForwardNetwork.Builder builder = new FeedForwardNetwork.Builder();

        List<String> allWeights = new ArrayList<>();
        List<double[]> allBiases = new ArrayList<>(); // still not implemented

        int width;
        String activation;

        for(Object jsonLayerObject : jsonLayers) {
            JSONObject layerObj = (JSONObject)jsonLayerObject;

            switch(LayerType.valueOf( layerObj.getString("layerType").toUpperCase() ) ) {
                case INPUT :
                        width = layerObj.getInt("width");
                        builder.addInputLayer(width);
                break;
                case FULLY_CONNECTED :
                        width = layerObj.getInt("width");
                        activation = layerObj.getString("activation").toUpperCase();
                         if (layerObj.has("weights")) {
                            String weights = layerObj.getString("weights");
                            allWeights.add(weights);
                         }
                        builder.addFullyConnectedLayer(width, ActivationType.valueOf(activation));
                break;
                case OUTPUT :
                        width = layerObj.getInt("width");
                        activation = layerObj.getString("activation").toUpperCase();
                        builder.addOutputLayer(width, ActivationType.valueOf(activation));
                        if (layerObj.has("weights")) {
                            String weights = layerObj.getString("weights");
                            allWeights.add(weights);
                        }
                break;
            }
        }

        // set loss function
        String lossFunction = jsonObj.getString("lossFunction");
        builder.lossFunction(LossType.valueOf(lossFunction));

        FeedForwardNetwork neuralNet = builder.build();

        return neuralNet;
    }
}