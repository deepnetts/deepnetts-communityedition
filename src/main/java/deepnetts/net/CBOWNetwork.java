package deepnetts.net;

import deepnetts.core.DeepNetts;
import deepnetts.net.layers.*;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.*;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.TensorBase;
//import deepnetts.tokenizers.Tokenizer;
import deepnetts.util.DeepNettsException;
import deepnetts.util.RandomGenerator;

public class CBOWNetwork extends NeuralNetwork<BackpropagationTrainer>{
    private transient TensorBase inputTensor;

    private CBOWNetwork() {
        super();
        this.setTrainer(new BackpropagationTrainer(this));
    }

    public static class Builder {

        private final CBOWNetwork network = new CBOWNetwork();

        private ActivationType defaultActivationType = ActivationType.TANH;

        private boolean setDefaultActivation = false;

        public CBOWNetwork.Builder addInputLayer(int layerWidth) {
            if (!this.network.getLayers().isEmpty()) {
                throw new DeepNettsException("Input layer must be the first layer in the network!");
            }
            InputLayer inLayer = new InputLayer(layerWidth);
            this.network.addLayer(inLayer);
            this.network.setInputLayer(inLayer);
            this.network.inputTensor = new Tensor1D(layerWidth);

            return this;
        }


//        public CBOWNetwork.Builder addTokenizer(Tokenizer tokenizer) {
//            this.network.tokenzier = tokenizer;
//            return this;
//        }


        public CBOWNetwork.Builder addInputLayer(int layerWidth, int batchSize) {
            if (!this.network.getLayers().isEmpty()) {
                throw new DeepNettsException("Input layer must be the first layer in the network!");
            }
            InputLayer inLayer = new InputLayer(layerWidth, batchSize);
            this.network.addLayer(inLayer);
            this.network.setInputLayer(inLayer);
            this.network.inputTensor = new Tensor2D(layerWidth, batchSize);

            return this;
        }


        public CBOWNetwork.Builder addCBOWEmbeddingLayer(int vocabSize, int embeddingDim) {
            CBOWEmbeddingLayer cbowEmbeddingLayer = new CBOWEmbeddingLayer(vocabSize, embeddingDim);
            this.network.addLayer(cbowEmbeddingLayer);

            return this;
        }


        public CBOWNetwork.Builder addFullyConnectedLayer(int layerWidth, ActivationType activationType) {
            FullyConnectedLayer layer = new FullyConnectedLayer(layerWidth, activationType);
            this.network.addLayer(layer);
            return this;
        }


        public CBOWNetwork.Builder addOutputLayer(int width, ActivationType activationType) {
            OutputLayer outputLayer = null;

            if (activationType.equals(ActivationType.SOFTMAX)) {
                outputLayer = new SoftmaxOutputLayer(width);
            } else {
                outputLayer = new OutputLayer(width, activationType);
            }

            this.network.setOutputLayer(outputLayer);
            this.network.addLayer(outputLayer);

            return this;
        }


        public CBOWNetwork.Builder lossFunction(LossType lossType) {
            LossFunction loss = null;

            switch (lossType) {
                case MEAN_SQUARED_ERROR:
                    loss = new MeanSquaredErrorLoss(this.network);
                    break;
                case CROSS_ENTROPY:
                    if (this.network.getOutputLayer().getWidth() == 1) {
                        loss = new BinaryCrossEntropyLoss(this.network);
                    } else {
                        loss = new CrossEntropyLoss(this.network);
                    }
                    break;
            }

            this.network.setLossFunction(loss);
            return this;
        }


        public CBOWNetwork.Builder randomSeed(long seed) {
            RandomGenerator.getDefault().initSeed(seed);
            return this;
        }


        public CBOWNetwork build() {

            AbstractLayer prevLayer = null;

            for (int i = 0; i < this.network.getLayers().size(); i++) {
                AbstractLayer layer = this.network.getLayers().get(i);
                layer.setPrevLayer(prevLayer);
                layer.setNetworkType(NetworkType.CBOW);
                if (prevLayer != null) {
                    prevLayer.setNextlayer(layer);
                }
                prevLayer = layer;
            }

            // init internal layer structures (weights, outputs, deltas etc. for each layer)
            for (AbstractLayer layer : this.network.getLayers()) {
                if (DeepNetts.getInstance().useCuda()) {
                    layer.setCudaHandles(network.cudaHandles); // if use cuda                    
                }                 
                if (DeepNetts.getInstance().isMultithreaded()) {
                    layer.setThreadPool(network.threadPool);
                }
                layer.init();
            }

            return this.network;
        }


    }

    public static CBOWNetwork.Builder builder(){
        return new CBOWNetwork.Builder();
    }
}
