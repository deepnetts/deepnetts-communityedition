package deepnetts.net.layers;

import deepnetts.core.DeepNetts;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.tensor.*;

import java.util.HashMap;

// NOTE ON BACKWARD PASS DESIGN!!!
// The standard DeepNetts backward convention stores dL/dz (pre-activation gradient) in this.deltas,
// and relies on the prev layer to recover dL/d(prev.outputs) by multiplying nextLayer.weights^T * nextLayer.deltas.
// LayerNorm has no weight matrix between input and output, so that pattern breaks down
// the prev layer cannot reconstruct dL/d(LayerNorm.inputs) from LayerNorm.deltas alone.
//
// To solve this cleanly, LayerNorm will introduce a dedicated field:
//
//   deltaInputs  — holds dL/d(LayerNorm.inputs), shape identical to inputs (embeddingDim, seqLen, batchSize)
//
// LayerNorm.backward() will:
//   1. Read nextLayer.deltaInputs to get dL/d(LayerNorm.outputs) — the upstream gradient
//      (this requires the next layer, e.g. FCL, to also compute and store its own deltaInputs), I still have to think if I want to do that ATM
//   2. Apply the LayerNorm Jacobian to produce dL/d(LayerNorm.inputs) and store it in this.deltaInputs - vector (dL/dOutputs) and a Jacobian (dOutputs/dLayerNorm.inputs)
//   3. Compute dGamma and dBeta from the upstream gradient and the cached xHat
//
// The prev layer of LayerNorm will read this.deltaInputs directly instead of
// doing nextLayer.weights^T * nextLayer.deltas, since there is no such weight matrix here.
//
// this.deltas will still be set (= this.deltaInputs for LINEAR activation) to avoid breaking
// any code that reads nextLayer.deltas generically.

public class LayerNorm extends AbstractLayer<TensorBase, TensorBase, TensorBase>{

    private int embeddingDim;
    private float epsilon = 0.00001f;
    private Tensor1D shiftWeights;
    public Tensor3D deltaInputs; // This is the Tensor that will hold the gradients of the loss fn with respect to the inputs to the LayerNorm, and it will be used by the prevLayer as is as deltas (might have to do activation prime first in that layer)
    private Tensor1D gradientsWeights;
    private Tensor1D gradientsBias;
    private HashMap<String, TensorBase> cache = new HashMap<>(); // I will use this to hold values needed for backward pass and these are calculated during forward pass

    public LayerNorm(int embeddingDim){
        super(ActivationType.LINEAR);
        this.embeddingDim = embeddingDim;
    }

    public LayerNorm(int embeddingDim, float epsilon){
        super(ActivationType.LINEAR);
        this.embeddingDim = embeddingDim;
        this.epsilon = epsilon;
    }

    @Override
    public void init() {
        this.inputs = prevLayer.getOutputs(); // (embeddingDim, sequenceLength, batchSize)
        int embeddingDim = this.inputs.shape().getDim(0);
        int sequenceLength = this.inputs.shape().getDim(1);
        int batchSize = this.inputs.shape().getDim(2);
        this.initWeights(embeddingDim);
        this.initBuffers(embeddingDim, sequenceLength, batchSize);

    }

    @Override
    public void forward() {
        this.forwardImpl.forward();
    }

    @Override
    public void backward(){
        this.zeroGrads();
        this.backwardImpl.backward();
    }

    @Override
    public void applyWeightChanges(){

        if (!isTrainable()) {
            return; // if layer is not trainable do not apply any changes
        }

        Tensors.copy(deltaWeights, prevDeltaWeights);
        Tensors.copy(deltaBiases, prevDeltaBiases);

        if (!DeepNetts.getInstance().useCuda()) {
            this.weights.add(deltaWeights);
            this.shiftWeights.add(deltaBiases);
        }
    }


    @Override
    public void initTransientFields(){
        this.forwardImpl = new LayerNorm.SingleThreadedForwardBatch();
        this.backwardImpl = new LayerNorm.SingleThreadedBackwardBatch();
    }

    private class SingleThreadedForwardBatch implements Forward {
        @Override
        public void forward(){
            int batchSize = inputs.shape().getDim(2);
            int contextLength = inputs.shape().getDim(1);
            int embeddingDim = inputs.shape().getDim(0);

            Tensor3D inputs3d = (Tensor3D) inputs;
            Tensor3D outputs3d = (Tensor3D) outputs;
            Tensor1D weights1d = (Tensor1D) weights;

            Tensor3D mean3d = new Tensor3D(1, contextLength, batchSize);
            Tensor3D rstd3d = new Tensor3D(1, contextLength, batchSize);


            for(int b = 0; b < batchSize; b++){
                for (int c = 0; c < contextLength; c++){
                    float acc = 0.0f;
                    float mean = 0.0f;
                    for (int e = 0; e < embeddingDim; e++){
                        acc += inputs3d.get(e, c, b);
                    }
                    mean = acc / embeddingDim;
                    mean3d.set(mean, 0, c, b);

                    float rstd;
                    float var;
                    float varAcc = 0;
                    for (int e = 0; e < embeddingDim; e++){
                        float x = inputs3d.get(e, c, b);
                        float xShifted = x - mean;
                        varAcc += xShifted * xShifted;
                    }
                    var = varAcc / embeddingDim;
                    rstd = (float) (1.0f / Math.sqrt(var + epsilon));
                    rstd3d.set(rstd, 0, c, b);
                    for (int e = 0; e < embeddingDim; e++){
                        float x = inputs3d.get(e, c, b);
                        float xNorm = (x - mean) * rstd;
                        float output = xNorm * weights1d.get(e) + shiftWeights.get(e);
                        outputs3d.set(output, e, c, b);
                    }
                }
            }

            cache.put("mean", (TensorBase) mean3d);
            cache.put("rstd", (TensorBase) rstd3d);
            outputs = outputs3d;
        }
    }

    private class SingleThreadedBackwardBatch implements Backward {
        @Override
        public void backward(){
            // THERE IS MORE EFFICIENT CLOSE FORM IMPLEMENTATION THAT WE CAN IMPLEMENT ONCE THIS WORKS
            // I JUST FEEL MORE CONFIDENT DEBUGGING THIS ATM
            // Based on the layer instance that is in front of LayerNorm we need to calculate dL/dLayerNorm.outputs
            // I would actually like to have this on nextLayer.dInputs but we will
            calculateOutputError(); // dL/dLayerNorm.outputs = this.deltas = dL/dNextLayerInputs
            int batchSize = deltas.shape().getDim(2);
            int contextLength = deltas.shape().getDim(1);
            int embeddingDim = deltas.shape().getDim(0);

            Tensor1D weights1d = (Tensor1D) weights;
            Tensor3D mean3d = (Tensor3D) cache.get("mean"); // (1, C, B)
            Tensor3D rstd3d = (Tensor3D) cache.get("rstd"); // (1, C, B)
            Tensor3D xNorm3d = new Tensor3D(embeddingDim, contextLength ,batchSize);
            Tensor3D inputs3d = (Tensor3D) inputs;
            Tensor3D xShifted3d = new Tensor3D(embeddingDim, contextLength ,batchSize);
            Tensor3D deltaxNorm3d = new Tensor3D(embeddingDim, contextLength, batchSize);
            Tensor3D deltaxShifted = new Tensor3D(embeddingDim, contextLength, batchSize);

            for(int b = 0; b < batchSize; b++) {
                for(int c = 0; c < contextLength; c++) {
                    for (int e = 0; e < embeddingDim; e++) {
                        float xShifted = inputs3d.get(e, c, b) - mean3d.get(0, c, b);
                        float xNorm = xShifted * rstd3d.get(0, c, b);
                        xNorm3d.set(xNorm, e, c, b);
                        xShifted3d.set(xShifted, e, c, b);
                    }
                }
            }
            // this.delta (E, C, B)
            Tensor3D deltas3d = (Tensor3D) deltas; // Gradients of output of layer norm w.r.t loss
            Tensor1D deltaWeights1D = (Tensor1D) deltaWeights;

            for(int b = 0; b < batchSize; b++) {
                for(int c = 0; c < contextLength; c++) {
                    float accRstd = 0;
                    for (int e = 0; e < embeddingDim; e++) {
                        // dxnorm = dout * weight
                        float dxNorm = deltas3d.get(e, c, b) * weights1d.get(e);
                        deltaxNorm3d.set(dxNorm, e, c, b);

                        accRstd += dxNorm * xShifted3d.get(e, c, b);
                    }

                    float accMean = 0;
                    for (int e = 0; e < embeddingDim; e++) {
                        float v = rstd3d.get(0, c, b) * (deltaxNorm3d.get(e, c, b) + accRstd * (-0.5f) * rstd3d.get(0, c, b) * rstd3d.get(0, c, b) * (2.0f / embeddingDim)  * xShifted3d.get(e, c, b));
                        accMean += v * (-1.0f / embeddingDim);
                        deltaxShifted.set(v, e, c, b);
                    }

                    for (int e = 0; e < embeddingDim; e++) {
                        // deltaInputs = deltaX
                        float deltaInput = deltaxShifted.get(e, c, b) + accMean;
                        deltaInputs.set(deltaInput, e, c, b); // They are ready to be taken by the other layer
                        // Shift weights and scale weight biases, I accumulate gradients for them over batch and sequence dimension
                        float dB = deltas3d.get(e, c, b);
                        gradientsBias.add(dB, e); // Shift weight gradients with respect to loss
                        float dW = deltas3d.get(e, c, b) * xNorm3d.get(e, c, b);
                        gradientsWeights.add(dW, e);
                    }
                }
            }
            // Loop where we calculate the net amount of change based on the optimizer rule
            // I do not like having optimizer do the work here
            // Easiest thing to do would be to move this kind of code to applyWeightChangesFn
            // Another thing that I do not like is that we hold these deltaWeights and deltaBiases
            // Optimizer should get a reference to weights and gradients associated with weights and then calculate the change which would be applied to
            // Why would optimizer even care about weights or biases, it should be occupied just with a notion of parameter
            for (int e = 0; e < embeddingDim; e++) {
                float dW = optimizer.calculateDeltaWeight(gradientsWeights.get(e), e);
                float dB = optimizer.calculateDeltaBias(gradientsBias.get(e), e);
                deltaWeights1D.set(dW, e);
                deltaBiases.set(dB, e);
            }
            deltaWeights = deltaWeights1D;
        }
    }

    public void initWeights(int embeddingDim){
        this.weights = new Tensor1D(embeddingDim); // Scale weights that we see as gamma in literature
        this.shiftWeights = new Tensor1D(embeddingDim); // Usually shown as beta in literature, and they are often initialized as all zeros

        this.weights.fill(1.0f); // As this is the standard way to initialize scaling weights for LayerNorm
    }

    public void initBuffers(int embeddingDim, int sequenceLength, int batchSize){
        this.outputs = new Tensor3D(embeddingDim, sequenceLength, batchSize);
        this.deltas =  new Tensor3D(embeddingDim, sequenceLength, batchSize); // dL/dLayerNorm.outputs
        this.deltaWeights = new Tensor1D(embeddingDim);
        // because of the current framework expectations I will put shiftWeights in deltaBiases
        this.deltaBiases = new Tensor1D(embeddingDim);
        // What will we do when layer has more weights like MHA Layer?
        this.gradientsBias = new Tensor1D(embeddingDim);
        this.gradientsWeights = new Tensor1D(embeddingDim);
    }

    private void zeroGrads(){
        this.gradients.fill(0);
        this.deltaWeights.fill(0);
        this.deltaBiases.fill(0);
        this.deltaInputs.fill(0);
        this.gradientsWeights.fill(0);
        this.gradientsBias.fill(0);
        this.deltas.fill(0);
    }

    public void calculateOutputError(){
        // It would be nice to skip all this stuff and just take deltas from NextLayer.dInputs
        if (nextLayer instanceof FullyConnectedLayer) {
            int embeddingDim = this.inputs.shape().getDim(0);
            int contextLength = this.inputs.shape().getDim(1);
            int batchSize = this.inputs.shape().getDim(2);
            int nextLayerWeightsRows = nextLayer.weights.shape().getDim(0);
            Tensor2D nextLayerWeights2d = (Tensor2D) nextLayer.weights;
            Tensor3D nextLayerDeltas3d = (Tensor3D) nextLayer.deltas;
            Tensor3D deltas3d = (Tensor3D) deltas;

            // nextLayer.deltas and nextLayer.weights
            // nextLayer.weights ^ T @ nextLayer.deltas
            // (2, 3) @ (3, 3, 5) = (2, 3, 5)

            for (int b = 0; b < batchSize; b++){
                for (int c = 0; c < contextLength; c++){
                    for (int e = 0; e < embeddingDim; e++) {
                        float acc = 0;
                        for (int k = 0; k < nextLayerWeightsRows; k++) {
                            acc += nextLayerWeights2d.get(k, e) * nextLayerDeltas3d.get(k, c, b);
                        }
                        deltas3d.set(acc, e, c, b);
                    }
                }
            }
            this.deltas = deltas3d;
        }
    }
}
