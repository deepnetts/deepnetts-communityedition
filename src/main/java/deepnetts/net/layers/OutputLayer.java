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
package deepnetts.net.layers;

import deepnetts.core.DeepNetts;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.layers.activation.MathFunctions;
import deepnetts.net.loss.LossType;
import deepnetts.net.weights.RandomWeights;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.TensorBase;
import deepnetts.tensor.Tensors;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.logging.Logger;

/**
 * Output layer of a neural network. It is always the last layer in a neural
 * network, and gives the final output of a network.
 *
 * @see AbstractLayer
 * @see Layer
 * @see NeuralNetwork
 *
 */
public class OutputLayer extends AbstractLayer<TensorBase, TensorBase, Tensor2D> {

    private static final long serialVersionUID = 4319240027730054207L;

    protected TensorBase outputErrors;
    protected final String[] labels;
    protected LossType lossType;

    protected transient boolean multithreaded = false;
    protected transient ArrayList<Callable<Void>> forwardTasks;
    protected transient ArrayList<Callable<Void>> backwardTasks;
//    private transient Forward forwardComputation; 
//    private transient Backward backwardComputation;

    private static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());

    private float singleOutInput;

    /**
     * Creates an instance of output layer with specified width (number of
     * outputs) and sigmoid activation function by default. Outputs are labeled
     * using generic names "Output1, 2, 3..."
     *
     * @param width layer width which represents number of network outputs
     */
    public OutputLayer(int width) {
        super(ActivationType.SIGMOID);
        this.width = width;
        this.height = 1;
        this.depth = 1;

        labels = new String[depth];
        // generate default output labels
        for (int i = 0; i < depth; i++) {
            labels[i] = "out" + i;
        }
    }

    /**
     * Creates an instance of output layer with specified width (number of
     * outputs) and specified activation function. Outputs are labeled using
     * generic names "Output1, 2, 3..."
     *
     * @param width layer width whic represents number of network outputs
     * @param actType activation function type for this layer
     */
    public OutputLayer(int width, ActivationType actType) {
        super(actType);
        this.width = width;
        this.height = 1;
        this.depth = 1;

        labels = new String[width];
        // generate enumerated class names from 1..n
        for (int i = 0; i < width; i++) {
            labels[i] = "Output" + i;
        }
    }

    /**
     * Creates an instance of output layer with specified width (number of
     * outputs) and linear activation function by default. Typically linear
     * activation is used for regression tasks, while sigmoid activation is used
     * for binary classification problems.
     *
     * @param outputLabels labels for network's outputs
     * @param actType activation function type for this layer
     */
    public OutputLayer(String[] outputLabels, ActivationType actType) {
        super(actType);
        this.width = outputLabels.length;
        this.height = 1;
        this.depth = 1;
        this.labels = outputLabels;
    }

    public final void setOutputErrors(final TensorBase outputErrors) {
        this.outputErrors = outputErrors;
    }

    public final TensorBase getOutputErrors() {
        return outputErrors;
    }

    public final LossType getLossType() {
        return lossType;
    }

    public void setLossType(LossType lossType) {
        this.lossType = lossType;
    }

    @Override
    public void init() {

        inputs = prevLayer.getOutputs();

        if (inputs instanceof Tensor1D) {
            batchSize = 1;
            batchMode = false;
            outputs = new Tensor1D(width);
            deltas = new Tensor1D(width);
            outputErrors = new Tensor1D(width);
        } else if (inputs instanceof Tensor2D) {
            batchMode = true;
            batchSize = ((Tensor2D) inputs).cols();
            outputs = new Tensor2D(width, batchSize);
            deltas = new Tensor2D(width, batchSize);
            outputErrors = new Tensor2D(width, batchSize);
        }

        int prevLayerWidth = prevLayer.getWidth();
        weights = new Tensor2D(width, prevLayerWidth);
        gradients = new Tensor2D(width, prevLayerWidth); // weight gradients
        deltaWeights = new Tensor2D(width, prevLayerWidth);
        prevDeltaWeights = new Tensor2D(width, prevLayerWidth);

        RandomWeights.xavier(weights.getValues(), prevLayerWidth, width);
        weights.createRowsCache();

        biases = new Tensor1D(width);
        deltaBiases = new Tensor1D(width);
        prevDeltaBiases = new Tensor1D(width);

        // TODO: koji metod randomizacije u zavisnosti od f-je transfera? Ili inicijalizuj preterniranim modelom
        RandomWeights.randomize(biases.getValues());

        if (!batchMode) {
            forwardImpl = new SingleThreadedForward(); // ovo ce biti defaultna implementacija
            backwardImpl = new SingleThreadedBackward();
        } else {
            forwardImpl = new SingleThreadedForwardBatch();
            backwardImpl = new SingleThreadedBackwardBatch();
        }

        initTransientFields();
    }

    @Override
    public void initTransientFields() {

        if (DeepNetts.getInstance().useCuda()) {
          //  forwardImpl = AcceleratorService.defaultProvider().createOutputForwardAcc(cudaHandles, this);
          //  backwardImpl = AcceleratorService.defaultProvider().createOutputBackwardAcc(cudaHandles, this);
        }

    }

    /**
     * This method implements forward pass for the output layer.
     *
     * Calculates weighted input and layer outputs using sigmoid function.
     */
    @Override
    public void forward() {
        inputs = prevLayer.getOutputs(); // is this neccesarry?

        forwardImpl.forward();

        // ovde nemamo multi threaded, ni gpu, bi batch - todo: dodati, da li ima smisla?
    }

    private class SingleThreadedForward implements Forward {

        @Override
        public void forward() {
            Tensor1D inputs1D = (Tensor1D) inputs;
            Tensor1D outputs1D = (Tensor1D) outputs;

            outputs = weights.matMul(inputs1D, outputs1D)
                    .add(biases)
                    .apply(activation);
        }
    }

    private class SingleThreadedForwardBatch implements Forward {

        @Override
        public void forward() {
            Tensor2D inputs2d = (Tensor2D) inputs;
            Tensor2D outputs2d = (Tensor2D) outputs;

            inputs2d.createColsCache();
            outputs = weights.matMul(inputs2d, outputs2d)
                    .add(biases)
                    .apply(activation);

            // ovo se sad ispod ne hvata, a ranije se koristilo u bce - proveri
            //  singleOutInput = outputs1d.get(0);
        }
    }

    // this should replace forward just make sure weigts cache is created before invoking this method
    private void forwardWithMatrixMultiplication1D() {
        Tensor1D inputs1d = (Tensor1D) inputs;
        Tensor1D outputs1d = (Tensor1D) outputs;

        //weights.createRowsCache(); // ovo raditi za inference samo jednom - u init?, kod treninga nakon izmene tezina - najbolje u apply weight changes
        outputs = weights.matMul(inputs1d, outputs1d)
                .add(biases);
        singleOutInput = outputs1d.get(0); // ovaj kao da se ne koristi nigde sad, proveri
        outputs.apply(activation);
    }

    /**
     * This method implements backward pass for the output layer.
     *
     * http://peterroelants.github.io/posts/neural_network_implementation_intermezzo01/
     * http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
     * http://neuralnetworksanddeeplearning.com/chap3.html
     */
    @Override
    public void backward() {

        if (DeepNetts.getInstance().useCuda()) {            
            deltas.copyFrom(outputErrors);             
            if (lossType == LossType.MEAN_SQUARED_ERROR) {                                   
              // ovde bi trebalo da ih deli sa n ako je mse loss i nije batch  - deli sa brojem outputa ako ih ima vise         
              TensorBase scaledDeltas = deltas.copy(); 
              float meanN = width;
              scaledDeltas.div(meanN);
              deltas.copyFrom(scaledDeltas);
          //    deltas.copyToGPU();  // ovo sadr radi cuda layer ne mor aovde
            }
        }
        backwardImpl.backward();
    }

//    @Override
//    public void backward() {
//        
//        if (!isTrainable()) return;
//        
//        if (!batchMode) {
//            Tensor1D inputs1D = (Tensor1D)inputs;
//            Tensor1D outputs1D = (Tensor1D)outputs;
//            Tensor1D deltas1D = (Tensor1D)deltas;
//            Tensor1D outputErrors1D = (Tensor1D)outputErrors;
//
//           if (!batchMode) {   // reset delta weights and deltaBiases to zero in each iteration if not in batch/minibatch mode
//               deltaWeights.fill(0);
//               deltaBiases.fill(0);
//           }
//
//           final float meanN = deltas.numElements(); // ali ovd emoze da bude num out * num vectors
//           outputErrors1D.div(meanN); // to je ono 1/n sto treba da deli kod mse da li i kod CE?
//           
//           for (int deltaIdx = 0; deltaIdx < deltas.numElements(); deltaIdx++) { // iterate all output neurons / deltas
//               if (lossType == LossType.MEAN_SQUARED_ERROR) {
//                   final float delta = outputErrors1D.get(deltaIdx) * activation.getPrime(outputs1D.get(deltaIdx));// / mean_n; // delta = e * dE/ds
//                   deltas1D.set(delta, deltaIdx);
//               } else if (activationType == ActivationType.SIGMOID && lossType == LossType.CROSS_ENTROPY) { // ovo samo za binary cross entropy, single sigmoid output
//                   deltas1D.set(outputErrors1D.get(deltaIdx), deltaIdx); // Bishop, pg. 231, eq.6.125, imenilac od dE/dy i izvod sigmoidne se skrate
//                   //65-
//               } // ... slucaj Cross entropy sa softmax je resen u SoftMaxLayer
//
//               for (int inIdx = 0; inIdx < inputs.numElements(); inIdx++) {
//                  final float regularization = (regL2 !=0 ?
//                                                    regL2 * 2 * weights.get(deltaIdx, inIdx) :
//                                                    regL1 * MathFunctions.absPrime(weights.get(deltaIdx, inIdx)));  
//
//                  // optimizovati jer sa istom deltom mnozimo sve elemente u inputu
//                  final float grad = deltas1D.get(deltaIdx) * inputs1D.get(inIdx) + regularization; // gradient dE/dw including regularization
//                  gradients.set(grad, deltaIdx, inIdx); // ovo samo cuva gradijente - da li ih negde koristimo?
//
//                  final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
//                  deltaWeights.add(deltaWeight, deltaIdx, inIdx); // add zbog batch moda!
//               }
//
//               final float deltaBias = optimizer.calculateDeltaBias(deltas1D.get(deltaIdx), deltaIdx);
//               deltaBiases.add(deltaBias, deltaIdx); // zasto je ovde add zar ne treba set??? zbog batcha, inace bi moglo set
//           }        
//        } else {
//            backwardBatch();
//        }
//        
//      //  System.out.println("DEltas:"+deltas.toString());
//    }
    private class SingleThreadedBackward implements Backward {

        @Override
        public void backward() {
            if (!isTrainable()) {
                return;
            }

            Tensor1D inputs1D = (Tensor1D) inputs;
            Tensor1D outputs1D = (Tensor1D) outputs;
            Tensor1D deltas1D = (Tensor1D) deltas;
            Tensor1D outputErrors1D = (Tensor1D) outputErrors;

            deltaWeights.fill(0);
            deltaBiases.fill(0);

            // ali mislim da ovo ne treba da radi ako je ce - proveri gradient check
            final float meanN = deltas.numElements(); // ali ovd emoze da bude num out * num vectors
            outputErrors1D.div(meanN); // to je ono 1/k sto treba da deli kod mse da li i kod CE?

            if (DeepNetts.getInstance().useVectorAPI()) {

                for (int deltaIdx = 0; deltaIdx < deltas.numElements(); deltaIdx++) { // iterate all output neurons / deltas
                    if (lossType == LossType.MEAN_SQUARED_ERROR) {
                        // ELEMENT WISE MNOZENJE - MOZE DA SE VEKTORIZUJEL ALI PRETHODNO MORA IZVOD
                        final float delta = outputErrors1D.get(deltaIdx) * activation.getPrime(outputs1D.get(deltaIdx));// / mean_n; // delta = e * dE/ds
                        deltas1D.set(delta, deltaIdx);
                    } else if (activationType == ActivationType.SIGMOID && lossType == LossType.CROSS_ENTROPY) { // ovo samo za binary cross entropy, single sigmoid output
                        deltas1D.set(outputErrors1D.get(deltaIdx), deltaIdx); // Bishop, pg. 231, eq.6.125, imenilac od dE/dy i izvod sigmoidne se skrate
                        //65-
                    } // ... slucaj Cross entropy sa softmax je resen u SoftMaxLayer
                }

                deltas1D.outerProduct(inputs1D, gradients);

                for (int deltaIdx = 0; deltaIdx < deltas.numElements(); deltaIdx++) { // iterate all output neurons / deltas
                    for (int inIdx = 0; inIdx < inputs.numElements(); inIdx++) {

                        // GDE SE PIRMENJUJE OVA REGULARIZACIJA??
                        final float regularization = (regL2 != 0
                                ? regL2 * 2 * weights.get(deltaIdx, inIdx)
                                : regL1 * MathFunctions.absPrime(weights.get(deltaIdx, inIdx)));

                        // optimizovati jer sa istom deltom mnozimo sve elemente u inputu
                        //final float grad = deltas1D.get(deltaIdx) * inputs1D.get(inIdx) + regularization; // gradient dE/dw including regularization
                        float grad = gradients.get(deltaIdx, inIdx); // ovo samo cuva gradijente - da li ih negde koristimo?

                        final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
                        deltaWeights.add(deltaWeight, deltaIdx, inIdx); // add zbog batch moda!
                    }

                    final float deltaBias = optimizer.calculateDeltaBias(deltas1D.get(deltaIdx), deltaIdx);
                    deltaBiases.add(deltaBias, deltaIdx); // zasto je ovde add zar ne treba set??? zbog batcha, inace bi moglo set
                }
            } else {
                for (int deltaIdx = 0; deltaIdx < deltas.numElements(); deltaIdx++) { // iterate all output neurons / deltas
                    if (lossType == LossType.MEAN_SQUARED_ERROR) {
                        final float delta = outputErrors1D.get(deltaIdx) * activation.getPrime(outputs1D.get(deltaIdx));// / mean_n; // delta = e * dE/ds
                        deltas1D.set(delta, deltaIdx);
                    } else if (activationType == ActivationType.SIGMOID && lossType == LossType.CROSS_ENTROPY) { // ovo samo za binary cross entropy, single sigmoid output
                        deltas1D.set(outputErrors1D.get(deltaIdx), deltaIdx); // Bishop, pg. 231, eq.6.125, imenilac od dE/dy i izvod sigmoidne se skrate
                        //65-
                    } // ... slucaj Cross entropy sa softmax je resen u SoftMaxLayer

                    for (int inIdx = 0; inIdx < inputs.numElements(); inIdx++) {
                        final float regularization = (regL2 != 0
                                ? regL2 * 2 * weights.get(deltaIdx, inIdx)
                                : regL1 * MathFunctions.absPrime(weights.get(deltaIdx, inIdx)));

                        // optimizovati jer sa istom deltom mnozimo sve elemente u inputu
                        final float grad = deltas1D.get(deltaIdx) * inputs1D.get(inIdx) + regularization; // gradient dE/dw including regularization
                        gradients.set(grad, deltaIdx, inIdx); // ovo samo cuva gradijente - da li ih negde koristimo?

                        final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
                        deltaWeights.add(deltaWeight, deltaIdx, inIdx); // add zbog batch moda!
                    }

                    final float deltaBias = optimizer.calculateDeltaBias(deltas1D.get(deltaIdx), deltaIdx);
                    deltaBiases.add(deltaBias, deltaIdx); // zasto je ovde add zar ne treba set??? zbog batcha, inace bi moglo set
                }
            }

        }
    }

    private class SingleThreadedBackwardBatch implements Backward {

        @Override
        public void backward() {
            if (!isTrainable()) {
                return;
            }

            Tensor2D inputs2D = (Tensor2D) inputs;
            Tensor2D deltas2D = (Tensor2D) deltas;
            Tensor2D outputs2D = (Tensor2D) outputs;
            Tensor2D outputErrors2D = (Tensor2D) outputErrors;

            // todo: inputs outputs, deltas and gradients must be 2d tesors - provide methods to get statistics of these
            // do it by layers and expochs  - for error analysys
            outputErrors2D.div(batchSize); //- - ali ovde treba deliti i sa brojem redova/outputa ak oih ima vise

            if (DeepNetts.getInstance().useVectorAPI()) {
                // iterate over batch rows
                for (int batchIdx = 0; batchIdx < inputs2D.cols(); batchIdx++) {
                    for (int deltaIdx = 0; deltaIdx < deltas2D.rows(); deltaIdx++) {

                        if (lossType == LossType.MEAN_SQUARED_ERROR) {
                            final float delta = outputErrors2D.get(deltaIdx, batchIdx) * activation.getPrime(outputs2D.get(deltaIdx, batchIdx)); // delta = e * dE/ds
                            deltas2D.set(delta, deltaIdx, batchIdx);
                        } else if (activationType == ActivationType.SIGMOID && lossType == LossType.CROSS_ENTROPY) { // ovo samo za binary cross entropy, single sigmoid output
                            deltas2D.set(outputErrors2D.get(deltaIdx, batchIdx), deltaIdx, batchIdx); // Bishop, pg. 231, eq.6.125, imenilac od dE/dy i izvod sigmoidne se skrate
                            //http://neuralnetworksanddeeplearning.com/chap3.html
                        }

                        // regularizaciju ne primenjuj na bias                
                        final float deltaBias = optimizer.calculateDeltaBias(deltas2D.get(deltaIdx, batchIdx), deltaIdx);
                        deltaBiases.add(deltaBias, deltaIdx);
                    }
                }

                Tensor2D inputsT = inputs2D.getTransposed();
                inputsT.createColsCache();
                deltas2D.createRowsCache();

                deltas2D.matMul(inputsT, gradients);

                // iterate over batch rows
                for (int colIdx = 0; colIdx < gradients.cols(); colIdx++) {
                    for (int rowIdx = 0; rowIdx < gradients.rows(); rowIdx++) {
                        final float regularization = (regL2 != 0
                                ? regL2 * 2 * weights.get(rowIdx, colIdx)
                                : regL1 * MathFunctions.absPrime(weights.get(rowIdx, colIdx)));

                        // problem je jer su deltas2D sve 0
                        //final float grad = deltas2D.get(deltaIdx, batchIdx) * inputs2D.get(inIdx, batchIdx) + regularization; // gradient dE/dw including regularization
                        //gradients.add(grad, deltaIdx, inIdx); // ovde zbog batcha mora add a ne set! @FIX: ispravi na drugim mestima ako ima
                        float grad = gradients.get(rowIdx, colIdx) + regularization;
                        gradients.set(grad, rowIdx, colIdx);

                        final float deltaWeight = optimizer.calculateDeltaWeight(grad, rowIdx, colIdx);
                        deltaWeights.add(deltaWeight, rowIdx, colIdx);

                    }
                }

            } else {

                // iterate over batch rows
                for (int batchIdx = 0; batchIdx < inputs2D.cols(); batchIdx++) {
                    for (int deltaIdx = 0; deltaIdx < deltas2D.rows(); deltaIdx++) {

                        if (lossType == LossType.MEAN_SQUARED_ERROR) {
                            final float delta = outputErrors2D.get(deltaIdx, batchIdx) * activation.getPrime(outputs2D.get(deltaIdx, batchIdx)); // delta = e * dE/ds
                            deltas2D.set(delta, deltaIdx, batchIdx);
                        } else if (activationType == ActivationType.SIGMOID && lossType == LossType.CROSS_ENTROPY) { // ovo samo za binary cross entropy, single sigmoid output
                            deltas2D.set(outputErrors2D.get(deltaIdx, batchIdx), deltaIdx, batchIdx); // Bishop, pg. 231, eq.6.125, imenilac od dE/dy i izvod sigmoidne se skrate
                            //http://neuralnetworksanddeeplearning.com/chap3.html
                        }

                        for (int inIdx = 0; inIdx < inputs2D.rows(); inIdx++) {
                            final float regularization = (regL2 != 0
                                    ? regL2 * 2 * weights.get(deltaIdx, inIdx)
                                    : regL1 * MathFunctions.absPrime(weights.get(deltaIdx, inIdx)));

                            // problem je jer su deltas2D sve 0
                            final float grad = deltas2D.get(deltaIdx, batchIdx) * inputs2D.get(inIdx, batchIdx) + regularization; // gradient dE/dw including regularization
                            gradients.add(grad, deltaIdx, inIdx); // ovde zbog batcha mora add a ne set! @FIX: ispravi na drugim mestima ako ima

                            final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
                            deltaWeights.add(deltaWeight, deltaIdx, inIdx);
                        }

                        // regularizaciju ne primenjuj na bias                
                        final float deltaBias = optimizer.calculateDeltaBias(deltas2D.get(deltaIdx, batchIdx), deltaIdx);
                        deltaBiases.add(deltaBias, deltaIdx);
                    }
                }
            }

        }
    }

    /**
     * Applies weight changes after one learning iteration or batch
     */
    @Override
    public void applyWeightChanges() {
        if (!isTrainable()) {
            return;
        }

 
        if (!DeepNetts.getInstance().useCuda()) {  
            // save current as prev delta weights (required for momentum)
            Tensors.copy(deltaWeights, prevDeltaWeights);            
            // apply(add) delta weights
            weights.add(deltaWeights);
            weights.createRowsCache(); // regenerate row cache for weights matrix

            // save current deltaBiases as prev delta biases
            Tensors.copy(deltaBiases, prevDeltaBiases);
            // apply(add) delta bias
            biases.add(deltaBiases);
        } 
          weights.createRowsCache(); // regenerate row cache for weights matrix
        

        if (batchMode) {    // for batch mode set all delta weights and biases to zero after applying changes. For online mode they are reseted in backward pass
            deltaWeights.fill(0);
            deltaBiases.fill(0);
        }

    }

    @Override
    public String toString() {
        return "Output Layer { width:" + width + ", activation:" + activationType.name() + "}";
    }

    private void forwardFromOneDimLayer() {
        outputs.copyFrom(biases);  // copy biases to all outputs, for summing

        if (!multithreaded) {
            if (inputs instanceof Tensor1D) {
                forwardForCellRange(0, outputs.numElements());
            } else if (inputs instanceof Tensor2D) { // batch mode is it known in inference?
                forwardForBatch(0, batchSize);
            }
        } else {
            try {
                threadPool.run(forwardTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage());
            }
        }
    }

    /**
     * Calculates forward pass for output cells in specified from to range.
     *
     * @param from
     * @param to
     */
    private void forwardForCellRange(final int from, final int to) {
        Tensor1D inputs1D = (Tensor1D) prevLayer.getOutputs();
        Tensor1D outputs1D = (Tensor1D) outputs;

        // bias is already copied to output in forwardFromOneDimLayer and forwardFrom3DLayer
        for (int outIdx = from; outIdx < to; outIdx++) {
            for (int inIdx = 0; inIdx < inputs.numElements(); inIdx++) { // zameniti mesta for da se out col brze vrti jer je na kraju weights
                final float weightedInput = weights.get(outIdx, inIdx) * inputs1D.get(inIdx);  // ovo treba da bude  weights[outRow, inRow] * input[inRow] // inCol postaje inRow
                outputs1D.add(weightedInput, outIdx);    // and add weighted sum to outputs
            }
            outputs1D.set(activation.getValue(outputs1D.get(outIdx)), outIdx); // replace with apply to all
        }
    }

    private void forwardForBatch(final int fromBatchIdx, final int toBatchIdx) {
        Tensor2D inputs2D = (Tensor2D) prevLayer.getOutputs(); // batch has 2d inputs
        Tensor2D outputs2D = (Tensor2D) outputs;

        // bias is already copied to output in forwardFromOneDimLayer and forwardFrom3DLayer
        for (int batchRowIdx = 0; batchRowIdx < inputs2D.rows(); batchRowIdx++) {
            for (int outIdx = 0; outIdx < outputs2D.cols(); outIdx++) {
                for (int inIdx = 0; inIdx < inputs.numElements(); inIdx++) { // zameniti mesta for da se out col brze vrti jer je na kraju weights
                    final float weightedInput = weights.get(outIdx, inIdx) * inputs2D.get(batchRowIdx, inIdx);  // ovo treba da bude  weights[outRow, inRow] * input[inRow] // inCol postaje inRow
                    outputs2D.add(weightedInput, batchRowIdx, outIdx);    // and add weighted sum to outputs
                }
                outputs2D.set(activation.getValue(outputs2D.get(batchRowIdx, outIdx)), batchRowIdx, outIdx); // replace with apply foreach batch
            }
        }
    }

    public float getSingleOutInput() {
        return singleOutInput;
    }

}
