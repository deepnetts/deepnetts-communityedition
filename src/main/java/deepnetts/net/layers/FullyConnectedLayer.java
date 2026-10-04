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
 * this program. If not, see <https://www.gnu.org/licenses/>.
 */
package deepnetts.net.layers;

import deepnetts.net.layers.activation.ActivationType;
import deepnetts.core.DeepNetts;
import deepnetts.net.Mode;
import deepnetts.net.layers.activation.ActivationFunction;
import deepnetts.net.layers.activation.MathFunctions;
import deepnetts.net.train.opt.Optimizer;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.net.weights.RandomWeights;
import deepnetts.util.CallableRangeConsumer;
import java.util.logging.Logger;
import deepnetts.util.DeepNettsException;
import deepnetts.util.DeepNettsThreadPool;
import deepnetts.util.RandomGenerator;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;
import deepnetts.tensor.Tensors;
import java.util.ArrayList;
import java.util.concurrent.Callable;

/**
 * Fully connected layer is used as a hidden layer in a neural network, and it
 * has a single row of units/nodes/neurons connected to all neurons in previous
 * and next layer. Previous layer can be input, fully connected, or flattened
 * layer, while next layer can be fully connected or output layer. This layer
 * calculates weighted sum of outputs from the previous layers (as matrix dot
 * product), and applies activation function to all that sum. Mathematical
 * formula is: Y = activation(W . X + B)
 *
 * where Y is output tensor (1D for single input or 2D for batch) W is a 2D
 * weights tensor X is input tensor (1D for single input or 2D for batch) B is a
 * 1D tensor of biases activation is an activation function
 *
 * @see ActivationType
 * @see ActivationFunction
 */
public class FullyConnectedLayer extends AbstractLayer<TensorBase, TensorBase, Tensor2D> { // input i output tenzor su uvek istih dimenzijaD

    // odlicno objasnjeneje matematike https://becominghuman.ai/understanding-neural-networks-2-the-math-of-neural-networks-in-3-equations-6085fd3f09df
    private static final long serialVersionUID = -8172689320815816927L;

    private static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());

    // inverted dropout notes
    // https://www.coursera.org/learn/deep-neural-network/lecture/eM33A/dropout-regularization
    // https://machinelearning.wtf/terms/inverted-dropout/    
    private boolean useDropout = false; //experimental ali deluje da radi!
    private float dropout = 0.2f;   // dropout probability
    private transient TensorBase dropouts;

    private transient boolean multithreaded = false;
    private transient boolean vectorAPI = false;
    private transient ArrayList<Callable<Void>> forwardTasks;
    private transient ArrayList<Callable<Void>> backwardTasks;

    /**
     * Creates an instance of fully connected layer with specified number of
     * neurons and ReLU activation function.
     *
     * @param layerSize a number of neurons in this layer / layer size (same as
     * number of outputs)
     */
    public FullyConnectedLayer(int layerSize) {
        super(ActivationType.RELU);

        if (layerSize > 0) {
            this.width = layerSize;
        } else {
            throw new DeepNettsException("Layer width must be non-zero, positive number");
        }

        this.height = 1;
        this.depth = 1;
    }

    /**
     * Creates an instance of a fully connected layer with specified width
     * (number of neurons) and activation function type.
     *
     * @param layerSize layer size / number of neurons in this layer
     * @param actType activation function type to use in this layer
     * @see ActivationFunctions
     */
    public FullyConnectedLayer(int layerSize, ActivationType actType) {
        super(actType);

        if (layerSize > 0) {
            this.width = layerSize;
        } else {
            throw new DeepNettsException("Layer width must be non-zero, positive number");
        }

        this.height = 1;
        this.depth = 1;
    }

    /**
     * Creates all internal data structures: inputs, weights, biases, outputs,
     * deltas, deltaWeights, deltaBiases prevDeltaWeights, prevDeltaBiases. Init
     * weights and biases. This method is called from network builder during
     * initialization
     */
    @Override
    public void init() {

        if (DeepNetts.getInstance().useVectorAPI()) {
            vectorAPI = true;
        }
        // check prev and next layers and throw exception if its illegall architecture
        if (!(prevLayer instanceof InputLayer
                || prevLayer instanceof FullyConnectedLayer
                || prevLayer instanceof FlattenLayer
                || prevLayer instanceof FlatEmbeddingLayer
                || prevLayer instanceof SkipGramEmbeddingLayer
                || prevLayer instanceof CBOWEmbeddingLayer
        )) {
            throw new DeepNettsException("Bad network architecture! Fully Connected Layer can be connected only to Input, FullyConnected, or Flatten layer as previous layer.");
        }

        if (nextLayer != null && !(nextLayer instanceof FullyConnectedLayer
                || nextLayer instanceof OutputLayer)) {
            throw new DeepNettsException("Bad network architecture! Fully Connected Layer can only be connected only to Fully Connected Layer or Output layer as next layer");
        }

        inputs = prevLayer.getOutputs();

        if (inputs instanceof Tensor1D) {
            batchSize = 1;
            batchMode = false;
            outputs = new Tensor1D(width);
            deltas = new Tensor1D(width);
            dropouts = new Tensor1D(width);
            weightedInput = new float[inputs.numElements()];
        } else if (inputs instanceof Tensor2D) {
            batchMode = true;
            batchSize = ((Tensor2D) inputs).cols();
            outputs = new Tensor2D(width, batchSize);
            deltas = new Tensor2D(width, batchSize);
            dropouts = new Tensor2D(width, batchSize);
        } else if (inputs instanceof Tensor4D) {
            // ovo je za batch kada ide na gpu
            batchMode = true;
            batchSize = ((Tensor4D) inputs).fourthDim(); // uzmi batch dimenziju
            outputs = new Tensor4D(batchSize, 1, width, 1); // da li je ovo u sustini isto kao i 2d batchSize, width???
            deltas = new Tensor4D(batchSize, 1, width, 1);
            dropouts = new Tensor4D(batchSize, 1, width, 1);
        }
        //kako inicijalizujem layer ako je prethodni flatten?
        // sta ako je input layer a nema vise dimenzija nego samo jednu?
        if (prevLayer instanceof FullyConnectedLayer || prevLayer instanceof FlattenLayer || prevLayer instanceof SkipGramEmbeddingLayer || prevLayer instanceof CBOWEmbeddingLayer || prevLayer instanceof FlatEmbeddingLayer || (prevLayer instanceof InputLayer && prevLayer.depth == 1)) { // && prevLayer.height == 1 ovo ako je prethodni 1d layer, odnosno ako je prethodni fully connected

            if (inputs instanceof Tensor1D) {
                weights = new Tensor2D(outputs.numElements(), inputs.numElements());
                deltaWeights = new Tensor2D(outputs.numElements(), inputs.numElements());  // da li ove upisujem negde uopste?
                gradients = new Tensor2D(outputs.numElements(), inputs.numElements());
                prevDeltaWeights = new Tensor2D(outputs.numElements(), inputs.numElements());

                // use he for relu
                if (activationType == ActivationType.RELU || activationType == ActivationType.LEAKY_RELU) {
                    RandomWeights.he(weights.getValues(), prevLayer.width); // idealno bi bilo da svaki neuron ima normalnu raspodelu tezina a ne cela matrics - proveri implemetacije
                } else {    // sigmoid tanh
                    RandomWeights.xavier(weights.getValues(), inputs.numElements(), outputs.numElements()); // outputs.size() is same as width
                    // RandomWeights.uniform(weights.getValues(), prevLayer.width); // outputs.size() is same as width
                }
            } else if (inputs instanceof Tensor2D || inputs instanceof Tensor4D) {
                weights = new Tensor2D(width, prevLayer.width);
                deltaWeights = new Tensor2D(width, prevLayer.width);
                gradients = new Tensor2D(width, prevLayer.width);
                prevDeltaWeights = new Tensor2D(width, prevLayer.width);

                if (activationType == ActivationType.RELU || activationType == ActivationType.LEAKY_RELU) {
                    RandomWeights.he(weights.getValues(), prevLayer.width); // idealno bi bilo da svaki neuron ima normalnu raspodelu tezina a ne cela matrics - proveri implemetacije
                } else {    // sigmoid tanh
                    RandomWeights.xavier(weights.getValues(), prevLayer.width, width); // outputs.size() is same as width
                    // RandomWeights.uniform(weights.getValues(), prevLayer.width); // outputs.size() is same as width
                }

                weights.createRowsCache();
            } // e sad ali kako kad je input 4D ?

            // vezuje tezine iz flatten layera direktno na ovaj layer - mislim da je ovo potpuno nepotrebno! jer flatten nikad ne koristi weights a i ne bi trebalo da dira weights iz sledeceg alyera
            if (prevLayer instanceof FlattenLayer) {
                prevLayer.weights = this.weights;
            }

        }

        biases = new Tensor1D(width);
        deltaBiases = new Tensor1D(width);
        prevDeltaBiases = new Tensor1D(width);

        if (activationType == ActivationType.RELU || activationType == ActivationType.LEAKY_RELU) {
            biases.fill(0.1f);
        } else {
            //Tensor.fill(biases, 0f);
            RandomWeights.gaussian(biases.getValues(), 0.1f, 0.05f);
        }

        // todo change after deserialization, neka bud eonaj koj ije deserijalizovan
        optimizer = Optimizer.create(OptimizerType.SGD, this); // default optimizer

        initTransientFields();
    }

    @Override
    public void initTransientFields() {
        //numThreads = getNumThreads(weights.numElements());
        numThreads = DeepNetts.getInstance().getMaxThreads();

        boolean oneDimLayer = (prevLayer instanceof FullyConnectedLayer || prevLayer instanceof FlattenLayer
                || (prevLayer instanceof InputLayer && prevLayer.getHeight() == 1 && prevLayer.getDepth() == 1));

        multithreaded = (numThreads > 1);

        if (multithreaded) {
            

            forwardTasks = new ArrayList<>();
            backwardTasks = new ArrayList<>();

            if (!batchMode) {
                int[] cellsPerThread = DeepNettsThreadPool.calculateCellsPerThread(width, numThreads);
                int from = 0, to = 0;

                for (int i = 0; i < numThreads; i++) {
                    from = to;
                    to = from + cellsPerThread[i];

                    if (oneDimLayer) { // forwardForCellRangeWithWeightCaching
                        CallableRangeConsumer ftask = new CallableRangeConsumer(from, to, this::forwardForCellRangeWithWeightCaching); // forwardForCellRange
                        forwardTasks.add(ftask);

                        CallableRangeConsumer btask = new CallableRangeConsumer(from, to, this::backwardTo1DLayerForRange);
                        backwardTasks.add(btask);
                    }
                }
            }
        }

        weights.createRowsCache();
        if (DeepNetts.getInstance().useCuda()) {
       //     forwardImpl = AcceleratorService.defaultProvider().createFullyConnectedForwardAcc(cudaHandles, this);
         //   backwardImpl = AcceleratorService.defaultProvider().createFullyConnectedBackwardAcc(cudaHandles, this); //new FullyConnectedBackwardCuda(cudaHandles, this);
        } else if (!multithreaded) { // single threaded
            if (!batchMode) { // inputs instanceof Tensor1D
                forwardImpl = new SingleThreadedMatrixForward(); // default forward
                backwardImpl = new SingleThreadedBackward();
            } else if (batchMode) { // inputs instanceof Tensor2D 
                forwardImpl = new SingleThreadedForwardBatch();
                backwardImpl = new SingleThreadedBackwardBatch();
            }
        } else { // multithreaded
            // ovo se ladno nigde nije testiralo ni pozvalo u svim testovima!
             if (!batchMode) {
                forwardImpl = new MultiThreadedForward();
                backwardImpl = new MultiThreadedBackward();
             } else { // multi threaded batch mode forward
                // throw new RuntimeException("Not implemented");
                forwardImpl = new MultiThreadedForwardBatch();
                backwardImpl = new MultiThreadedBackwardBatch();
             }
        }
    }

    @Override
    public void forward() {

        if (prevLayer.getOutputs() instanceof Tensor1D tensor1D) { // batchMode == true
            inputs = tensor1D; // posto je u input layer setuje ceo tenzor kao ulaz a on je istovremeno i izlaz 
        } else if (prevLayer.getOutputs() instanceof Tensor2D tensor2D) { // batch mode cpu
            inputs = tensor2D;
        } else if (prevLayer.getOutputs() instanceof Tensor4D tensor4D) { // batch mode gpu
            inputs = tensor4D;
        }

        forwardImpl.forward();

        // perform dropout        
        if (mode == Mode.TRAIN && useDropout) { // kako ovo radi za gpu?
            dropout();
        }

    }

    private void dropout() {
        final float keepProb = 1 - dropout;
        float[] dropoutValues = dropouts.getValues();
        for (int i = 0; i < dropouts.numElements(); i++) {
            final float p = RandomGenerator.getDefault().nextFloat();
            final float val = (p > dropout ? 1f / (1-p)  : 0f); // inverted dropout
            //dropouts.set(val, i);
            dropoutValues[i] = val;
            //outputs.set(val * outputs.get(i), i);
            // multiply with keepProb for inverted dropout see 	https://machinelearning.wtf/terms/inverted-dropout/
            // outputs.multiply(val, i); // ovo isto bolje da vektorizujem  - mislim d abi trebalo samo val, i)
            // trebalo bi mnoziti sa keep prob kao inverted dropout    - a i mislim da ovo nije dobro             
        }
        outputs.multiplyElementWise(dropouts);
    }

    /**
     * Calculate forward pass, when the previous layer is also 1-dim.
     */
//    private void forwardFromOneDimLayer() { // cpu forward
//        // ova metoda ce se zameniti sa ucitavanjem implementacije u initu i pozivanje forwarad od implementacije
//        if (!multithreaded) { // single threaded
//            if (inputs instanceof Tensor1D) { // not batch !batchMode
//                //forwardForCellRange(0, outputs.numElements());
//                // forwardWithMatrixMultiplication1D();
//                forwardComputation.forward();
//            } else if (inputs instanceof Tensor2D) { // batchMode==true
//               // forwardBatch(); // ovaj ce da pomnozi matrice i vec ddodaje biase
//               forwardComputation.forward();
//            }
//        } else { // multithreaded
    ////            outputs.copyFrom(biases);
////            try {
////                threadPool.run(forwardTasks);
////            } catch (InterruptedException ex) {
////                LOG.warning(ex.getMessage());
////            
//            forwardComputation.forward();
//            // multithreaded batch forward ce da podeli delove batch-a po threadovima po potrebi, ako u batchu ima vise od x elemenata
//        }
//    }

    // OUT = Weights * Input  gde je input row vector a weight 2d matrica dimenzja[inputs, outputs]
    // Y = W . X + B
    // vidi slajdove duboko ucenje sa univerziteta slajd 16
    // https://becominghuman.ai/understanding-neural-networks-2-the-math-of-neural-networks-in-3-equations-6085fd3f09df    
    /**
     * Calculates forward pass for output cells in specified range.
     *
     * @param from starting output cell
     * @param to ending output cell (exclusively)
     */
    private void forwardForCellRange(final int from, final int to) {
        // bias is already copied to output in forwardFromOneDimLayer and forwardFrom3DLayer
        // ovo je mnozenje matrice vektorom M . v - zapravo jednog dela za koji je zaduzen thread

        // trebalo bi promeniti da bud ekompatibilno sa cuda level 2 blas matrix vector multilication
        // TODO: kreiraj output ili weightedInputSum kao array buffer kao niz tako da ne mora da da radi add i get    
        // takodje i inputs da bude arr
        // y = f(W . x + b)
        // promeniti da bude kao u tf i pt: y = f(x . W + b)
        // input u batch-u , i jdan niz kao row buffer za input, i tezine da se smanji broj izracunavanja za poziciju samo getRow, a mozda i get Col - to se posle moze lepo vektorizovati
        final int inputsSize = inputs.numElements();
        float weightedInputSum;

        Tensor1D inputs1d = (Tensor1D) inputs;
        Tensor1D outputs1d = (Tensor1D) outputs;

        // outputs = weights.dotProd(input, outputs).add(biases).apply(af);
        for (int outIdx = from; outIdx < to; outIdx++) {
            weightedInputSum = 0;
            for (int inIdx = 0; inIdx < inputsSize; inIdx++) {
                weightedInputSum += weights.get(outIdx, inIdx) * inputs1d.get(inIdx);
            }
            outputs1d.add(weightedInputSum, outIdx);    // and add weighted sum to outputs - mora add zbog biasa, ili dodati bias u weighted sum prethodno!!!! pa ne mora uvek add nego samo jednom
        }
        activation.apply(outputs, from, to);
    }

    transient float[] weightedInput;

    private void forwardForCellRangeWithWeightCaching(final int from, final int to) {
      //  Tensor1D inputs1d = (Tensor1D) inputs;
        Tensor1D outputs1d = (Tensor1D) outputs;

        final int inputsSize = inputs.numElements();
        final float[] inputVect = inputs.getValues(); // podrazumeva se da je 1d tenzor
        float weightedInputSum;

        for (int outIdx = from; outIdx < to; outIdx++) {
            final float[] weightsRow = weights.getRowsCache(outIdx);
            weightedInputSum = 0;
            // ovde se desava 65% izracunavanja za inference
            for (int inIdx = 0; inIdx < inputsSize; inIdx++) {
                //  weightedInput[inIdx]= weightsRow[inIdx] * inputVect[inIdx];
                weightedInputSum += weightsRow[inIdx] * inputVect[inIdx];
            }
//            for (int inIdx = 0; inIdx < inputsSize; inIdx++) { 
//                weightedInputSum += weightedInput[inIdx];
//            }            

            outputs1d.add(weightedInputSum, outIdx);    // and add weighted sum to outputs - mora add zbog biasa, ili dodati bias u weighted sum prethodno!!!! pa ne mora uvek add nego samo jednom
        }
        activation.apply(outputs, from, to);
    }

    //   private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private void forwardForCellRangeWithVectorAPI() {
        Tensor1D inputs1d = (Tensor1D) inputs;
        Tensor1D outputs1d = (Tensor1D) outputs;

        final int inputsSize = inputs.numElements();
        final float[] inputVect = inputs.getValues();
        float weightedInputSum;

        for (int outIdx = 0; outIdx < outputs1d.numElements(); outIdx++) {
            final float[] weightsRow = weights.getRowsCache(outIdx);
            weightedInputSum = 0;

            //   int vectorLength = SPECIES.length();
            int inIdx = 0;
//            for (; inIdx <= inputsSize - vectorLength; inIdx += vectorLength) {
//                var v1 = FloatVector.fromArray(SPECIES, weightsRow, inIdx); // ove sve iskesiraj u prvom prolazu - ne unapred, tako da ih spoljni loop vec ima
//                var v2 = FloatVector.fromArray(SPECIES, inputVect, inIdx);
//                weightedInputSum += v1.mul(v2).reduceLanes(VectorOperators.ADD);
//            }

            // ovo treba razbiti u dve kako bi se auto vektorizovalo
            for (; inIdx < inputsSize; inIdx++) {
                weightedInputSum += weightsRow[inIdx] * inputVect[inIdx];
            }

            outputs1d.add(weightedInputSum, outIdx);    // and add weighted sum to outputs - mora add zbog biasa, ili dodati bias u weighted sum prethodno!!!! pa ne mora uvek add nego samo jednom
        }
        //activation.apply(outputs);
        outputs.apply(activation);
    }

    private class MultiThreadedForwardBatch implements Forward {

       MultiThreadedForwardBatch() {
            //weights.setThreadPool(threadPool); 
       }
        
        @Override
        public void forward() { 
            weights.setThreadPool(threadPool); // ovo bi trebalo u konstruktoru ali pucaju testovi, jer se nesto ne inicijalizuje
            
            Tensor2D inputs2d = (Tensor2D) inputs;
            Tensor2D outputs2d = (Tensor2D) outputs;
   
            inputs2d.createColsCache();// ovo samo ako je u traininig modu
            outputs = weights.matMul(inputs2d, outputs2d) // ovo da radu multi threaded matMulMt
                    .add(biases) // ovde da radi auto vektorizaciju
                    .apply(activation);                        
        }
    }

    // default implementation of single threaded single vector matrix forward 
    // replacement of forwardWithMatrixMultiplication1D
    private class SingleThreadedMatrixForward implements Forward {

        @Override
        public void forward() {
            Tensor1D inputs1d = (Tensor1D) inputs;
            Tensor1D outputs1d = (Tensor1D) outputs;

            //weights.createRowsCache(); // ovo raditi za inference samo jednom - u init?, kod treninga nakon izmene tezina - najbolje u apply weight changes
            outputs = weights.matMul(inputs1d, outputs1d)
                    .add(biases)
                    .apply(activation);
        }
    }

    private class SingleThreadedForwardBatch implements Forward {
        
        @Override
        public void forward() {
            Tensor2D inputs2d = (Tensor2D) inputs; // ove ubaci kao generice daih ne castuje u svakom prolazu  ili u konstruktoru
            Tensor2D outputs2d = (Tensor2D) outputs;

          //  if (mode == Mode.TRAIN) {
                inputs2d.createColsCache();// ovo samo ako je u traininig modu
         //   }

            outputs = weights.matMul(inputs2d, outputs2d)
                    .add(biases)
                    .apply(activation);

        }
    }

    private class MultiThreadedForward implements Forward {
        // init forward tasks here
        @Override
        public void forward() {
            outputs.copyFrom(biases);
            try {
                threadPool.run(forwardTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage());
            }
        }
    }

    // ovo je single threaded i 1d, napravi da moze matrix matrix odnosno batch i multithreaded
    private void forwardWithMatrixMultiplication1D() {
        Tensor1D inputs1d = (Tensor1D) inputs;
        Tensor1D outputs1d = (Tensor1D) outputs;

        //weights.createRowsCache(); // ovo raditi za inference samo jednom - u init?, kod treninga nakon izmene tezina - najbolje u apply weight changes
        // ubaci threadove u dotProd 
        outputs = weights.matMul(inputs1d, outputs1d)
                .add(biases)
                .apply(activation);
    }

    // mutithreaded strausen algo
    // dodaj forwardForBatchRange(final int fromBatchIdx, final int toBatchIdx)
    private void forwardBatch() {

        // trebalo bi promeniti da bud ekompatibilno sa cuda level 2 blas matrix vector multilication
        // y = f(W . x + b)
        // promeniti da bude kao u tf i pt: y = f(x . W + b)

        Tensor2D inputs2d = (Tensor2D) inputs;
        Tensor2D outputs2d = (Tensor2D) outputs;

        // ovde treba mnoziti mtrice umesto ovog ispod i onda primeniti act func
        // prvo da proradi single threaded onda multi threaded
        inputs2d.createColsCache();// ovo samo ako je u traininig modu
        outputs = weights.matMul(inputs2d, outputs2d)
                .add(biases)
                .apply(activation);

    }

    // En-1 = WnT * En   transponuj matricu tezina i pomnozi sa gresama i deltama funkcija izvoda - to je za backward
    @Override
    public void backward() {

        if (!isTrainable()) {
            return;
        }

        if (DeepNetts.getInstance().useCuda()) {    
            backwardImpl.backward();         
            return;
        }
        
        // step 1 : propagate errors from next layer
        if (inputs instanceof Tensor1D) { // single threaded single vector backward
            // ovo prilagoditi batch-u ovako samo ako nije batch
            // ovo id eu varijantu bez batcha - a napraviti i varijantu za batchom - back phase 1
            Tensor1D nextDeltas1D = (Tensor1D) nextLayer.deltas;
            Tensor1D deltas1D = (Tensor1D) deltas;
            Tensor1D outputs1D = (Tensor1D) outputs;
            final Tensor2D nextLayerWeights = (Tensor2D) nextLayer.weights;

            // ovo sad nije u batthc modu
            if (!batchMode) { // if in online mode (not batch mode) reset deltaWeights and deltaBiases to zeros
                deltaWeights.fill(0);
                deltaBiases.fill(0);
            }

            deltas.fill(0); // reset current deltas


            // STEP 1. propagate and sum weighted deltas (outputGRadients) from the next layer (which can be output or fully connected) to calculate deltas for this layer
            for (int deltaIdx = 0; deltaIdx < deltas.numElements(); deltaIdx++) {   // for every neuron/delta in this layer
                for (int ndIdx = 0; ndIdx < nextLayer.deltas.numElements(); ndIdx++) { // iterate all neurons/deltas from next layer
                    final float weightedDelta = nextLayerWeights.get(ndIdx, deltaIdx) * nextDeltas1D.get(ndIdx); // @check: proveri da li su ove dobre
                    deltas1D.add(weightedDelta, deltaIdx); // calculate weighted sum of deltas from the next layer
                }

                final float delta = deltas1D.get(deltaIdx) * activation.getPrime(outputs1D.get(deltaIdx));
                deltas1D.set(delta, deltaIdx);
            } // end sum weighted deltas from next layer

            // apply dropout if needed
            if (useDropout) { // should be used only during training
                deltas.multiplyElementWise(dropouts); // e ovo treba isto prilagoditi batch-u
            }
        } else if (inputs instanceof Tensor2D) { // single threaded batch mode
            // propagate deltas from next layer through weights to this layer for batch
            Tensor2D nextDeltas2D = (Tensor2D) nextLayer.deltas;
            Tensor2D deltas2D = (Tensor2D) deltas;
            Tensor2D outputs2D = (Tensor2D) outputs;
            final Tensor2D nextLayerWeights = (Tensor2D) nextLayer.weights;
            deltas.fill(0); // reset current deltas

            // todo : nextDeltas pomnoziti matricno sa weights i tako propagoras delte u prethodni sloj
            for (int batchIdx = 0; batchIdx < outputs2D.cols(); batchIdx++) {
                for (int deltaIdx = 0; deltaIdx < deltas2D.rows(); deltaIdx++) {   // for every neuron/delta in this layer
                    for (int ndIdx = 0; ndIdx < nextDeltas2D.rows(); ndIdx++) { // iterate all neurons/deltas from next layer
                        final float weightedDelta = nextLayerWeights.get(ndIdx, deltaIdx) * nextDeltas2D.get(ndIdx, batchIdx); // @check: proveri da li su ove dobre
                        deltas2D.add(weightedDelta, deltaIdx, batchIdx); // calculate weighted sum of deltas from the next layer
                    }

                    final float delta = deltas2D.get(deltaIdx, batchIdx) * activation.getPrime(outputs2D.get(deltaIdx, batchIdx));
                    deltas2D.set(delta, deltaIdx, batchIdx);
                } // end sum weighted deltas from next layer            
            }
        } 

        // STEP 2. calculate delta weights // if previous layer is Dense (2D weights matrix) - optimize
        if ((prevLayer instanceof FullyConnectedLayer || prevLayer instanceof FlattenLayer)
                || ((prevLayer instanceof InputLayer) && (prevLayer.height == 1 && prevLayer.depth == 1))) { // ili 1d Input Layer, dodati uslov
            backwardSingleLayer();
        }
    }

    private void backwardSingleLayer() {
        // aha ovde dodaj GPU!
        if (!multithreaded) { // single threaded
            if (inputs instanceof Tensor1D) { // single vector
                // backwardTo1DLayerForRange(0, deltas.numElements());
                backwardImpl.backward();
            } else if (inputs instanceof Tensor2D) { // batch
                //  backwardTo1DLayerBatch();
                backwardImpl.backward();
            }
        } else { // multithreaded
            backwardImpl.backward();
            // ovde nedostaje batch varijanta, 
        }
    }

    private class MultiThreadedBackward implements Backward {

        @Override
        public void backward() {
            try {
                threadPool.run(backwardTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage());
            }
        }
    }
    

        private class MultiThreadedBackwardBatch implements Backward {
            private final Tensor2D inputs2DTransposed;
            private final Tensor2D deltas2D = (Tensor2D) deltas;
            private final Tensor2D gradients2D = (Tensor2D) gradients;


            public MultiThreadedBackwardBatch() {
                Tensor2D inputs2D = (Tensor2D) prevLayer.getOutputs();
                inputs2DTransposed = new Tensor2D(inputs2D.cols(), inputs2D.rows());
                deltas2D.setThreadPool(threadPool);
            }
         
         
          
            @Override
            public void backward() {
                Tensor2D inputs2D = (Tensor2D) prevLayer.getOutputs(); // mora ovde zato sto mu se input tenzor menmja ako je posle input layera

                inputs2D.transposeInto(inputs2DTransposed);
                deltas2D.createRowsCache();
                inputs2DTransposed.createColsCache();
                deltas2D.matMul(inputs2DTransposed, gradients2D);
               
                // ovo vec radi optimizer
                deltaWeights.copyFrom(gradients2D);
                //deltaWeights.multiply(-learningRate); // ovo ubaci u optimizer a gradjente ubacuj direktno u delta weights, skrati                
                optimizer.calculateDeltaWeight(deltaWeights);

                deltas2D.sumByColInto(deltaBiases);
                // deltaBiases.multiply(-learningRate);                                          
                optimizer.calculateDeltaBias(deltaBiases);
            }

    }
    
    

    // batch implemntiran
    private void backwardTo1DLayerBatch() {
        Tensor2D inputs2d = (Tensor2D) inputs;
        Tensor2D deltas2d = (Tensor2D) deltas;

        for (int batchIdx = 0; batchIdx < inputs2d.cols(); batchIdx++) {
            for (int deltaIdx = 0; deltaIdx < deltas2d.rows(); deltaIdx++) {
                for (int inIdx = 0; inIdx < inputs2d.rows(); inIdx++) {

                    float grad = deltas2d.get(deltaIdx, batchIdx) * inputs2d.get(inIdx, batchIdx);// + regularization; // gradient dE/dw + L2 regularization

                    if (regL2 != 0 && regL1 != 0) {
                        final float regularization = (regL2 != 0
                                ? // ovo uvek primenjuje regularizaciju, da li tako treba, da li to hocu?
                                regL2 * 2 * weights.get(deltaIdx, inIdx)
                                : regL1 * MathFunctions.absPrime(weights.get(deltaIdx, inIdx)));
                        grad += regularization;
                    }

                    gradients.add(grad, deltaIdx, inIdx); // not used at the moment
                    final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
                    deltaWeights.add(deltaWeight, deltaIdx, inIdx); // add zbog batch moda!
                }

                // regularizaciju ne primenjuj na bias
                final float deltaBias = optimizer.calculateDeltaBias(deltas2d.get(deltaIdx, batchIdx), deltaIdx);
                deltaBiases.add(deltaBias, deltaIdx);
            }
        }
    }

    // replacement for backwardTo1DLayerBatch
    private class SingleThreadedBackwardBatch implements Backward {

        @Override
        public void backward() {
            Tensor2D inputs2d = (Tensor2D) inputs; // ovo moze samo jednom u konstruktoru da uradi
            Tensor2D deltas2d = (Tensor2D) deltas;

            if (vectorAPI) {

                Tensor2D inputsT = inputs2d.getTransposed(); // ovaj cemo transpose into
                inputsT.createColsCache();
                deltas2d.createRowsCache();

                deltas2d.matMul(inputsT, gradients);

                // hajde sad i delta weights vektorizovan a ne ovako sa for ispodsla
                
                for (int deltaIdx = 0; deltaIdx < deltas2d.rows(); deltaIdx++) {
                    for (int inIdx = 0; inIdx < inputs2d.rows(); inIdx++) {

                        //float grad = deltas2d.get(deltaIdx, batchIdx) * inputs2d.get(inIdx, batchIdx);// + regularization; // gradient dE/dw + L2 regularization
                        float grad = gradients.get(deltaIdx, inIdx);

                        if (regL2 != 0 && regL1 != 0) {
                            final float regularization = (regL2 != 0
                                    ? // ovo uvek primenjuje regularizaciju, da li tako treba, da li to hocu?
                                    regL2 * 2 * weights.get(deltaIdx, inIdx)
                                    : regL1 * MathFunctions.absPrime(weights.get(deltaIdx, inIdx)));
                            grad += regularization;
                        }

                        gradients.set(grad, deltaIdx, inIdx); // not used at the moment
                        final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
                        deltaWeights.add(deltaWeight, deltaIdx, inIdx); // add zbog batch moda!
                    }
                }

                for (int batchIdx = 0; batchIdx < inputs2d.cols(); batchIdx++) {
                    for (int deltaIdx = 0; deltaIdx < deltas2d.rows(); deltaIdx++) {
                        // regularizaciju ne primenjuj na bias
                        final float deltaBias = optimizer.calculateDeltaBias(deltas2d.get(deltaIdx, batchIdx), deltaIdx);
                        deltaBiases.add(deltaBias, deltaIdx);
                    }
                }

            } else {
                for (int batchIdx = 0; batchIdx < inputs2d.cols(); batchIdx++) {
                    for (int deltaIdx = 0; deltaIdx < deltas2d.rows(); deltaIdx++) {
                        for (int inIdx = 0; inIdx < inputs2d.rows(); inIdx++) {

                            float grad = deltas2d.get(deltaIdx, batchIdx) * inputs2d.get(inIdx, batchIdx);// + regularization; // gradient dE/dw + L2 regularization

                            if (regL2 != 0 && regL1 != 0) {
                                final float regularization = (regL2 != 0
                                        ? // ovo uvek primenjuje regularizaciju, da li tako treba, da li to hocu?
                                        regL2 * 2 * weights.get(deltaIdx, inIdx)
                                        : regL1 * MathFunctions.absPrime(weights.get(deltaIdx, inIdx)));
                                grad += regularization;
                            }

                            gradients.add(grad, deltaIdx, inIdx); // not used at the moment
                            final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
                            deltaWeights.add(deltaWeight, deltaIdx, inIdx); // add zbog batch moda!
                        }

                        // regularizaciju ne primenjuj na bias
                        final float deltaBias = optimizer.calculateDeltaBias(deltas2d.get(deltaIdx, batchIdx), deltaIdx);
                        deltaBiases.add(deltaBias, deltaIdx);
                    }
                }
            }

        }
    }

    // replacement for  backwardTo1DLayerForRange(0, deltas.numElements());
    private class SingleThreadedBackward implements Backward {

        @Override
        public void backward() {
            Tensor1D inputs1d = (Tensor1D) inputs;
            Tensor1D deltas1d = (Tensor1D) deltas;

            if (vectorAPI) {

                deltas1d.outerProduct(inputs1d, gradients);

                for (int deltaIdx = 0; deltaIdx < deltas.numElements(); deltaIdx++) { // this iterates neurons (weights depth)
                    for (int inIdx = 0; inIdx < inputs.numElements(); inIdx++) {

                        //float grad = deltas1d.get(deltaIdx) * inputs1d.get(inIdx);// + regularization; // gradient dE/dw + L2 regularization
                        float grad = gradients.get(deltaIdx, inIdx);

                        if (regL2 != 0 && regL1 != 0) {
                            final float regularization = (regL2 != 0
                                    ? // ovo uvek primenjuje regularizaciju, da li tako treba, da li to hocu?
                                    regL2 * 2 * weights.get(deltaIdx, inIdx)
                                    : regL1 * MathFunctions.absPrime(weights.get(deltaIdx, inIdx)));
                            grad += regularization;
                        }

                        gradients.set(grad, deltaIdx, inIdx); // not used anywhere
                        final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
                        deltaWeights.add(deltaWeight, deltaIdx, inIdx); // add zbog batch moda!
                    }

                    // regularizaciju ne primenjuj na bias
                    final float deltaBias = optimizer.calculateDeltaBias(deltas1d.get(deltaIdx), deltaIdx);
                    deltaBiases.add(deltaBias, deltaIdx);
                }

            } else {

                for (int deltaIdx = 0; deltaIdx < deltas.numElements(); deltaIdx++) { // this iterates neurons (weights depth)
                    for (int inIdx = 0; inIdx < inputs.numElements(); inIdx++) {

                        float grad = deltas1d.get(deltaIdx) * inputs1d.get(inIdx);// + regularization; // gradient dE/dw + L2 regularization

                        if (regL2 != 0 && regL1 != 0) {
                            final float regularization = (regL2 != 0
                                    ? // ovo uvek primenjuje regularizaciju, da li tako treba, da li to hocu?
                                    regL2 * 2 * weights.get(deltaIdx, inIdx)
                                    : regL1 * MathFunctions.absPrime(weights.get(deltaIdx, inIdx)));
                            grad += regularization;
                        }

                        gradients.set(grad, deltaIdx, inIdx); // not used anywhere
                        final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
                        deltaWeights.add(deltaWeight, deltaIdx, inIdx); // add zbog batch moda!
                    }

                    // regularizaciju ne primenjuj na bias
                    final float deltaBias = optimizer.calculateDeltaBias(deltas1d.get(deltaIdx), deltaIdx);
                    deltaBiases.add(deltaBias, deltaIdx);
                }

            }

        }
    }

    private void backwardTo1DLayerForRange(final int from, final int to) {
        Tensor1D inputs1d = (Tensor1D) inputs;
        Tensor1D deltas1d = (Tensor1D) deltas;

        for (int deltaIdx = from; deltaIdx < to; deltaIdx++) { // this iterates neurons (weights depth)
            for (int inIdx = 0; inIdx < inputs.numElements(); inIdx++) {

                float grad = deltas1d.get(deltaIdx) * inputs1d.get(inIdx);// + regularization; // gradient dE/dw + L2 regularization

                if (regL2 != 0 && regL1 != 0) {
                    final float regularization = (regL2 != 0
                            ? // ovo uvek primenjuje regularizaciju, da li tako treba, da li to hocu?
                            regL2 * 2 * weights.get(deltaIdx, inIdx)
                            : regL1 * MathFunctions.absPrime(weights.get(deltaIdx, inIdx)));
                    grad += regularization;
                }

                gradients.set(grad, deltaIdx, inIdx); // not used anywhere
                final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
                deltaWeights.add(deltaWeight, deltaIdx, inIdx); // add zbog batch moda!
            }

            // regularizaciju ne primenjuj na bias
            final float deltaBias = optimizer.calculateDeltaBias(deltas1d.get(deltaIdx), deltaIdx);
            deltaBiases.add(deltaBias, deltaIdx);
        }
    }

    @Override
    public void applyWeightChanges() {
        if (!isTrainable()) {
            return; // if layer is not trainable do not apply any changes 
        }
        
        // ovo mozda treba izbaciti jer se vec desava u cuda layeru, ili je mozda ostavljeno ovo zbog cpu
//        if (batchMode) { // podeli Delta weights sa velicinom batch-a
//            deltaWeights.div(batchSize); // i ne samo to nego ih treba sabrati, po kojoj dimenziji - po batchu, row, odnosno nultoj osi
//            deltaBiases.div(batchSize); // 
//        }

        Tensors.copy(deltaWeights, prevDeltaWeights); // save as prev delta weight
        Tensors.copy(deltaBiases, prevDeltaBiases);

        if (!DeepNetts.getInstance().useCuda()) {
            weights.add(deltaWeights);
            biases.add(deltaBiases);
        }

        weights.createRowsCache();

        if (batchMode) {
            deltaWeights.fill(0);
            deltaBiases.fill(0);
        }

//        if (DeepNetts.getInstance().useCuda()) {
//            weights.copyToGPU();
//            biases.copyToGPU();
//        }

    }

    public float getDropout() {
        return dropout;
    }

    public void setDropout(float dropout) {
        this.dropout = dropout;
        this.useDropout = true;
    }

    @Override
    public String toString() {
        return "Fully Connected Layer { width:" + width + " activation:" + activationType.name() + "}";
    }

}
