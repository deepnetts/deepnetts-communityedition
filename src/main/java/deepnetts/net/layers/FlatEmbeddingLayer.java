package deepnetts.net.layers;

import deepnetts.core.DeepNetts;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.weights.RandomWeights;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.TensorBase;
import deepnetts.tensor.Tensors;
import deepnetts.util.DeepNettsException;

import java.util.logging.Logger;

// Embedding Layer used for Feed Forward Net Language Model presented in Bengio et al. 2003
// TODO fix indexing - determine precisely which dimension is supposed to be batch dim and which one is the other one
public class FlatEmbeddingLayer extends AbstractLayer<TensorBase, TensorBase, Tensor2D>{
    // Add serial field here

    private int vocabSize;
    private int embeddingDim;

    private static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());

    // weights = vocabSize x embeddingDim
    public FlatEmbeddingLayer(int vocabSize, int embeddingDim){
        super(ActivationType.LINEAR); // Will this cause a problem? This layer really needs no activation
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
    }

    @Override
    public void init(){
        this.assertCorrectArchitectureConfig();
        this.inputs = prevLayer.getOutputs();
        // Extract dimensions of inputs - batch size, sequence length for this layer
        // Check if the batch dim is no of rows or no of columns
        // I actually think that for this framework batch dimension is the last dimension so I shell prob change this
        int batchSize = prevLayer.batchSize;
        int sequenceLength = prevLayer.width;
        this.width = sequenceLength * this.embeddingDim; // I am adding this because Fully Connected layer needs it to instantiate it's weight matrix
        this.initWeights(this.vocabSize, this.embeddingDim);
        this.initBuffers(batchSize, sequenceLength, this.embeddingDim);
        this.initTransientFields();
    }


    @Override
    public void forward(){
        this.forwardImpl.forward();
    }


    @Override
    public void backward(){
        this.zeroGrads(); // In PyTorch Optimizer Class zeros out the gradient
        this.backwardImpl.backward();
    }


    @Override
    public void applyWeightChanges(){
        // This method performs weight update
        if (!isTrainable()) {
            return; // if layer is not trainable do not apply any changes
        }

        Tensors.copy(deltaWeights, prevDeltaWeights); // Not necessary for all optimizers I would say

        if (!DeepNetts.getInstance().useCuda()) {
            weights.add(deltaWeights);
        }
    }


    @Override
    public void initTransientFields(){
        // Only single threaded implementation for now
        // Is there really a need for separate classes based on batch/no batch?
        // Instead of no batch we can have batch = 1 as default
        this.forwardImpl = new SingleThreadedForwardBatch();
        this.backwardImpl = new SingleThreadedBackwardBatch();
    }


    // Nested classes that implement Forward and Backward interfaces are used for running forward and backward methods
    // Abstract layer holds transient fields forward of type Forward and backward of type Backward which we need to initialize
    // I suggest renaming these to forwardEngine and backwardEngine or something similar because this is confusing at the moment
    // These could also be declared static classes
    private class SingleThreadedForwardBatch implements Forward {
        @Override
        public void forward(){
            int batchSize = inputs.shape().getDim(1);
            int contextLength = inputs.shape().getDim(0);

            Tensor2D inputs2d = (Tensor2D) inputs; // Will this create a new copy of inputs?
            Tensor2D outputs2d = (Tensor2D) outputs; // (contextLength * embeddingDim) x batchSize  where embeddingDim is class attribute

            for(int i = 0; i < batchSize; i++){
                for(int j = 0; j < contextLength; j++){
                    int tokenId = (int) inputs2d.get(j, i);
                    for(int e = 0; e < embeddingDim; e++){
                        float featureValue = weights.get(tokenId, e);
                        int row = j * embeddingDim + e;
                        outputs2d.set(featureValue, row, i);
                    }
                }
            }

            outputs = outputs2d; // Do I need this really? I guess I do but I have to check

        }
    }


    private class SingleThreadedBackwardBatch implements Backward {
        @Override
        public void backward(){
            // Error coming from the layer above this one
            // I do not think we shell store nextLayer.deltas as this will become inefficient
            // More inclined towards receiving that in backward function
            Tensor2D deltasNextLayer2d = (Tensor2D) nextLayer.deltas; // nextLayerWidth x batchSize
            Tensor2D nextLayerWeights = (Tensor2D) nextLayer.weights; // nextLayerWidth x (sequenceLength x embeddingDim)
            // deltas = W_next_layer^T @ deltas_next_layer (contextLength * embeddingDim) x batchSize but I will note save them since I do not need them for further use by other layers
            Tensor2D inputs2d = (Tensor2D) inputs;
            int sequenceLength = inputs2d.rows();

            // Are gradients zero initialized, additionally they have to be zeroed out on each new iteration

            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < sequenceLength; j++) {
                    int tokenId = (int) inputs2d.get(j, i);
                    for (int e = 0; e < embeddingDim; e++) {
                        int flatIdx = j * embeddingDim + e;
                        int innerDim = nextLayerWeights.rows();
                        float grad = 0;
                        for (int k = 0; k < innerDim; k++) {
                            grad += nextLayerWeights.get(k, flatIdx) * deltasNextLayer2d.get(k, i);
                        }
                        gradients.add(grad, tokenId, e);
                        float dW = optimizer.calculateDeltaWeight(grad, tokenId, e); // I am not sure about why this needs indices
                        deltaWeights.add(dW, tokenId, e);
                    }
                }
            }
        }
    }

    // For the purposes of weight initialization I will create a helper method that I will call inside init() method
    private void initWeights(int vocabSize, int embeddingDim){
        this.weights = new Tensor2D(vocabSize, embeddingDim);
        RandomWeights.normal(this.weights.getValues());
    }

    // I am encapsulating logic for checking the previous layer compatibility inside this method
    private void assertCorrectArchitectureConfig() throws DeepNettsException {
        boolean condition = !(this.prevLayer instanceof InputLayer);
        if (condition) {
            throw new DeepNettsException("Bad network architecture! Flat Embedding Layer can only be connected only to Input Layer where Input Layer is the one preceding it!");
        }
    }

    // Inside this helper method I will initialize all of the buffers needed for BP and FP
    private void initBuffers(int batchSize, int contextLength, int embeddingDim){
        this.outputs = new Tensor2D(contextLength * embeddingDim, batchSize);
        this.gradients = new Tensor2D(this.vocabSize, this.embeddingDim); // Are these zero initialized? I do not thinks so
        this.deltaWeights = new Tensor2D(this.vocabSize, this.embeddingDim); // This is the net amount of change to be applied to weights (-lambda * gradients)
        this.prevDeltaWeights = new Tensor2D(this.vocabSize, this.embeddingDim); // I guess this is used for certain type of optimizers and it does not seem like it is always needed so this can be unnecessary allocation
        this.prevDeltaWeights.fill(0);
    }

    private void zeroGrads(){
        this.gradients.fill(0);
        this.deltaWeights.fill(0);
    }

    public void setVocabSize(int vocabSize){
        this.vocabSize = vocabSize;
    }

    public void setEmbeddingDim(int embeddingDim) {
        this.embeddingDim = embeddingDim;
    }

    public int getVocabSize(){
        return this.vocabSize;
    }

    public int getEmbeddingDim(){
        return this.embeddingDim;
    }

    @Override
    public String toString() {
        return "Flat Embedding Layer { vocab size:" + this.vocabSize + " embedding dimension:" + this.embeddingDim + "}";
    }
}
