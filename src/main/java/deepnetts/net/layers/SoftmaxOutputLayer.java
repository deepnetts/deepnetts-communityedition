/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 */
package deepnetts.net.layers;

import deepnetts.accl.spi.AcceleratorService;
import deepnetts.accl.spi.ActivationVectorizationProvider;
import deepnetts.accl.spi.ActivationVectorizationService;
import deepnetts.core.DeepNetts;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.layers.activation.MathFunctions;
import deepnetts.net.weights.RandomWeights;
import deepnetts.util.CallableRangeConsumer;
import deepnetts.util.DeepNettsThreadPool;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor4D;
import java.util.ArrayList;

/**
 * Output layer with softmax activation function.
 *
 * @author Zoran Sevarac
 */
public class SoftmaxOutputLayer extends OutputLayer {  
    private static final long serialVersionUID = 609777460047517229L;
    private static ActivationVectorizationProvider vectorizationImpl = ActivationVectorizationService.defaultProvider();;


    //private transient boolean multithreaded = false;
    public SoftmaxOutputLayer(int size) {
        super(size, ActivationType.SOFTMAX);
    }

    @Override
    public void init() {
        inputs = prevLayer.getOutputs();
        if (inputs instanceof Tensor1D) {
            outputs = new Tensor1D(width);
            outputErrors = new Tensor1D(width);
            deltas = new Tensor1D(width);

            // tensor width/cols je koliko ima neurona u prethodnom FC lejeru  - pretpostavka je da moze samo FC lejer da bude iza
            weights = new Tensor2D(outputs.numElements(), inputs.numElements());
            deltaWeights = new Tensor2D(outputs.numElements(), inputs.numElements());
            gradients = new Tensor2D(outputs.numElements(), inputs.numElements());
            prevDeltaWeights = new Tensor2D(outputs.numElements(), inputs.numElements());

            RandomWeights.xavier(weights.getValues(), inputs.numElements(), outputs.numElements());
        } else if (inputs instanceof Tensor2D) {
            Tensor2D inputs2d = (Tensor2D) prevLayer.getOutputs();
            batchMode = true;
            batchSize = inputs2d.cols();
            outputs = new Tensor2D(width, batchSize);
            outputErrors = new Tensor2D(width, batchSize);
            deltas = new Tensor2D(width, batchSize);

            Tensor2D outputs2d = (Tensor2D) outputs;
            // tensor width/cols je koliko ima neurona u prethodnom FC lejeru  - pretpostavka je da moze samo FC lejer da bude iza
            weights = new Tensor2D(outputs2d.rows(), inputs2d.rows());
            deltaWeights = new Tensor2D(outputs2d.rows(), inputs2d.rows());
            gradients = new Tensor2D(outputs2d.rows(), inputs2d.rows());
            prevDeltaWeights = new Tensor2D(outputs2d.rows(), inputs2d.rows());

            RandomWeights.xavier(weights.getValues(), inputs2d.rows(), outputs2d.rows());
            weights.createRowsCache();
        } else if (inputs instanceof Tensor4D) {
            Tensor4D inputs4d = (Tensor4D) prevLayer.getOutputs();
            batchMode = true;
            batchSize = inputs4d.fourthDim();
            outputs = new Tensor4D(batchSize, 1, width, 1);
            outputErrors = new Tensor4D(batchSize, 1, width, 1);
            deltas = new Tensor4D(batchSize, 1, width, 1);

            Tensor4D outputs4d = (Tensor4D) outputs;
            // tensor width/cols je koliko ima neurona u prethodnom FC lejeru  - pretpostavka je da moze samo FC lejer da bude iza
            weights = new Tensor2D(outputs4d.rows(), inputs4d.rows());
            deltaWeights = new Tensor2D(outputs4d.rows(), inputs4d.rows());
            gradients = new Tensor2D(outputs4d.rows(), inputs4d.rows());
            prevDeltaWeights = new Tensor2D(outputs4d.rows(), inputs4d.rows());

            RandomWeights.xavier(weights.getValues(), inputs4d.rows(), outputs4d.rows());
        }

        biases = new Tensor1D(width);
        deltaBiases = new Tensor1D(width);
        prevDeltaBiases = new Tensor1D(width);

        //RandomWeights.randomize(biases);
        biases.fill(0.0f); // ovo je za relu dobro, ali da li ovde hocu relu???
//        RandomWeights.gaussian(biases, 0.1f, 0.05f);

//        RandomWeights.randomize(biases);
        initTransientFields();
    }

    @Override
    public void initTransientFields() {
        //   numThreads = getNumThreads(weights.numElements()); // broj threadova treba da bude uskladjen sa velicinom layera
        numThreads = DeepNetts.getInstance().getMaxThreads();

        multithreaded = (numThreads > 1);
        if (multithreaded) {
            int[] cellsPerThread = DeepNettsThreadPool.calculateCellsPerThread(width, numThreads);

            forwardTasks = new ArrayList<>();
            backwardTasks = new ArrayList<>();

            int from = 0, to = 0;

            for (int i = 0; i < numThreads; i++) {
                from = to;
                to = from + cellsPerThread[i];

                CallableRangeConsumer ftask = new CallableRangeConsumer(from, to, this::forwardForCellRange);
                forwardTasks.add(ftask);

                CallableRangeConsumer btask = new CallableRangeConsumer(from, to, this::backwardForCellRange);
                backwardTasks.add(btask);

            }
        }

        weights.createRowsCache();

        if (DeepNetts.getInstance().useCuda()) {
            //forward = new SoftMaxForwardCuda(cudaHandles, this);
            forwardImpl = AcceleratorService.defaultProvider().createSoftmaxOutputForwardAcc(cudaHandles, this); //new SoftMaxForwardCuda(cudaHandles, this);
            //backward = new SoftMaxBackwardCuda(cudaHandles, this);
            backwardImpl = AcceleratorService.defaultProvider().createSoftmaxOutputBackwardAcc(cudaHandles, this);
        } else if (!multithreaded) { // single threaded
            if (!batchMode) { // inputs instanceof Tensor1D
                forwardImpl = new SingleThreadedForward(); // default forward
                backwardImpl = new SingleThreadedBackward();
            } else if (batchMode) { // inputs instanceof Tensor2D , resi i slucaj za Tensor4D to je na GPU
                forwardImpl = new SingleThreadedForwardBatch();
                backwardImpl = new SingleThreadedBackwardBatch();
            }
        } else { // multithreaded
            if (!batchMode) {
                forwardImpl = new MultiThreadedForward();
                backwardImpl = new MultiThreadedBackward();
            } else {
                forwardImpl = new MultiThreadedForwardBatch();
                backwardImpl = new MultiThreadedBackwardBatch();  
            }
        }
    }

    /**
     * Forward pass for the output layer, calculates layer outputs using softmax
     * function.
     */
    @Override
    public void forward() {

        //1D - inline
        if (inputs instanceof Tensor1D) {
            outputs.copyFrom(biases); // da li ovo icemu sluzi? da li je potrebno???

            // CUDA
            if (DeepNetts.getInstance().useCuda()) {
                forwardImpl.forward(); // ovaj scenario necu nikad ni koristiit - sa za dev
                return;
            }
            // 4D - FORWARD BATCH 
        } else if (inputs instanceof Tensor4D) {
            // zasto ovde nema copy from biases
            // CUDA
            if (DeepNetts.getInstance().useCuda()) {
                forwardImpl.forward();
                return;
            }
        }

        forwardImpl.forward();
    }
    
    // 
    private void forwardForCellRange(final int from, final int to) {
        Tensor1D inputs1D = (Tensor1D) prevLayer.getOutputs();
        Tensor1D outputs1D = (Tensor1D) outputs;

        float weightedInputSum;
        for (int outIdx = from; outIdx < to; outIdx++) { // for all outputs in from, to range
            weightedInputSum = 0; // mogao bih i ovde da izvucem nizove pa da primeni auto vektorizaciju a u tenzorima samo da ih drzim
            for (int inIdx = 0; inIdx < inputs.numElements(); inIdx++) { // iterate all inputs
                weightedInputSum += weights.get(outIdx, inIdx) * inputs1D.get(inIdx);
            }
            outputs1D.add(weightedInputSum, outIdx); // add weighted sum of inputs  - mora add a ne set  zbog biasa!!!!!! ili gore u weighted sum dodaj bias a onda ovde mozes set
        }
    }

    // zamena za forwardWithMatrixMultiplication1D
    private class SingleThreadedForward implements Forward {

        @Override
        public void forward() {
            // find max weightedSum
            float maxWs = Float.NEGATIVE_INFINITY;

            outputs.copyFrom(biases); // ovo se gazi 

            Tensor1D inputs1d = (Tensor1D) inputs;
            Tensor1D outputs1d = (Tensor1D) outputs;

            //weights.createRowsCache(); // ovo raditi za inference samo jednom - u init?, kod treninga nakon izmene tezina - najbolje u apply weight changes
            outputs = weights.matMul(inputs1d, outputs1d)
                             .add(biases);

            applySoftmax(outputs1d, maxWs); // maxWs se ne koristi
        }

    }

    private void forwardWithMatrixMultiplication1D() {
        Tensor1D inputs1d = (Tensor1D) inputs;
        Tensor1D outputs1d = (Tensor1D) outputs;

        //weights.createRowsCache(); // ovo raditi za inference samo jednom - u init?, kod treninga nakon izmene tezina - najbolje u apply weight changes
        outputs = weights.matMul(inputs1d, outputs1d)
                         .add(biases);
    }

    private class MultiThreadedForward implements Forward {

        @Override
        public void forward() {
            // find max weightedSum
            float maxWs = Float.NEGATIVE_INFINITY;

            Tensor1D outputs1D = (Tensor1D) outputs;
            outputs.copyFrom(biases);

            try {
                threadPool.run(forwardTasks);
            } catch (InterruptedException ex) {
                //LOG.warning(ex.getMessage());
            }

            // find max weighted sum
            for (int outIdx = 0; outIdx < outputs.numElements(); outIdx++) {                    // for all neurons in this layer
                if (outputs1D.get(outIdx) > maxWs) {
                    maxWs = outputs1D.get(outIdx);
                }
            }

            // calculate outputs and denominator sum (use max for numerical stability)
            float denSum = 0;
            for (int outIdx = 0; outIdx < outputs.numElements(); outIdx++) {
                outputs1D.set((float) Math.exp(outputs1D.get(outIdx) - maxWs), outIdx); // maxWs used for numerical stability
                denSum += outputs1D.get(outIdx);
            }

            outputs.div(denSum); // scale all outputs to sum to 1                    
        }
    }
    
  private class MultiThreadedForwardBatch implements Forward {

        @Override
        public void forward() {
            Tensor2D inputs2d = (Tensor2D) inputs;
            Tensor2D outputs2d = (Tensor2D) outputs;
            weights.setThreadPool(threadPool); // ovo ne treba ovde nego negde ranije u initu mozda

            inputs2d.createColsCache(); // @FIX: ovo se mozda moze izbaciti
            outputs = weights.matMul(inputs2d, outputs2d) // ovo da radu multi threaded matMulMt
                             .add(biases);
            
            applySoftmaxBatch(inputs2d, outputs2d);
        }
    }

      //  ovo je realno samo preimenovana single theraded batch implemenatcija
  
      
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

            Tensor2D inputs2D = (Tensor2D) prevLayer.getOutputs();
            
            inputs2D.transposeInto(inputs2DTransposed);
            deltas2D.createRowsCache();
            inputs2DTransposed.createColsCache();
            deltas2D.matMul(inputs2DTransposed, gradients2D);
            
            deltaWeights.copyFrom(gradients2D);
            deltaWeights.multiply(-learningRate);
            
            deltas2D.sumByColInto(deltaBiases);
            deltaBiases.multiply(-learningRate);                                          
        }
    }
  
  
    private class SingleThreadedForwardBatch implements Forward {

        @Override
        public void forward() {
            Tensor2D inputs2D = (Tensor2D) prevLayer.getOutputs();
            Tensor2D outputs2D = (Tensor2D) outputs;

            inputs2D.createColsCache(); // ovo samo u trening modu

            outputs = weights.matMul(inputs2D, outputs2D)
                             .add(biases);// moras da ih dodas na svaku kolonu! da broadcastujes po kolonama / batches

//            System.out.println("weights: "+weights);
//            System.out.println("inputs: "+inputs2D);
//            System.out.println("outputs: "+outputs2D);            
            
            applySoftmaxBatch(inputs2D, outputs2D);
       //      System.out.println("outputs after softmax: "+outputs2D);
//            for (int batchIdx = 0; batchIdx < inputs2D.cols(); batchIdx++) {
//                float maxWs = Float.NEGATIVE_INFINITY;
//                for (int outIdx = 0; outIdx < outputs2D.rows(); outIdx++) {                    // for all neurons in this layer
//                    if (outputs2D.get(outIdx, batchIdx) > maxWs) {
//                        maxWs = outputs2D.get(outIdx, batchIdx);
//                    }
//                }
//
//                // calculate outputs and denominator sum (use max for numerical stability)
//                float denSum = 0;
//                for (int outIdx = 0; outIdx < outputs2D.rows(); outIdx++) {
//                    outputs2D.set((float) Math.exp(outputs2D.get(outIdx, batchIdx) - maxWs), outIdx, batchIdx); // maxWs used for numerical stability
//                    denSum += outputs2D.get(outIdx, batchIdx);
//                }
//
//                for (int outIdx = 0; outIdx < outputs2D.rows(); outIdx++) {
//                    final float newVal = outputs2D.get(outIdx, batchIdx) / denSum;
//                    outputs2D.set(newVal, outIdx, batchIdx);
//                }
//                //outputs.div(denSum); // scale all outputs to sum to 1             
//            }
        }
    }

    private void forwardBatch() {
        Tensor2D inputs2D = (Tensor2D) prevLayer.getOutputs();
        Tensor2D outputs2D = (Tensor2D) outputs;

        inputs2D.createColsCache(); // ovo samo u trening modu

        outputs = weights.matMul(inputs2D, outputs2D)
                         .add(biases); // moras da ih dodas na svaku kolonu! da broadcastujes po kolonama / batches

        for (int batchIdx = 0; batchIdx < inputs2D.cols(); batchIdx++) {
            float maxWs = Float.NEGATIVE_INFINITY;
            for (int outIdx = 0; outIdx < outputs2D.rows(); outIdx++) {                    // for all neurons in this layer
                if (outputs2D.get(outIdx, batchIdx) > maxWs) {
                    maxWs = outputs2D.get(outIdx, batchIdx);
                }
            }

            // calculate outputs and denominator sum (use max for numerical stability)
            float denSum = 0;
            for (int outIdx = 0; outIdx < outputs2D.rows(); outIdx++) {
                outputs2D.set((float) Math.exp(outputs2D.get(outIdx, batchIdx) - maxWs), outIdx, batchIdx); // maxWs used for numerical stability
                denSum += outputs2D.get(outIdx, batchIdx);
            }

            for (int outIdx = 0; outIdx < outputs2D.rows(); outIdx++) {
                final float newVal = outputs2D.get(outIdx, batchIdx) / denSum;
                outputs2D.set(newVal, outIdx, batchIdx);
            }
            //outputs.div(denSum); // scale all outputs to sum to 1  ovo se sad radi iznad           
        }

        //  ali ostadose negativni!!!
        /*float weightedInputSum;
            for (int batchIdx = 0; batchIdx < inputs2D.cols(); batchIdx++) {
                for (int outIdx = 0; outIdx < outputs2D.cols(); outIdx++) { // for all outputs in from, to range
                    weightedInputSum = 0; // mogao bih i ovde da izvucem nizove pa da primeni auto vektorizaciju a u tenzorima samo da ih drzim
                    for (int inIdx = 0; inIdx < inputs2D.cols(); inIdx++) { // iterate all inputs
                        weightedInputSum += weights.get(outIdx, inIdx) * inputs2D.get(batchIdx, inIdx);
                    }
                    outputs2D.add(weightedInputSum, batchIdx, outIdx); // add weighted sum of inputs  - mora add a ne set  zbog biasa!!!!!! ili gore u weighted sum dodaj bias a onda ovde mozes set
                }
            }
         */
    }

    // ovaj za multithreaded foerward batch - ili bolje uraditi paralelizaciju mnozenja matrica?
    private void forwardBatch(final int fromBatchIdx, final int toBatchIdx) {
        Tensor2D inputs2D = (Tensor2D) prevLayer.getOutputs();
        Tensor2D outputs2D = (Tensor2D) outputs;

        float weightedInputSum;
        for (int batchRowIdx = 0; batchRowIdx < inputs2D.rows(); batchRowIdx++) {
            for (int outIdx = 0; outIdx < outputs2D.cols(); outIdx++) { // for all outputs in from, to range
                weightedInputSum = 0; // mogao bih i ovde da izvucem nizove pa da primeni auto vektorizaciju a u tenzorima samo da ih drzim
                for (int inIdx = 0; inIdx < inputs2D.cols(); inIdx++) { // iterate all inputs
                    weightedInputSum += weights.get(outIdx, inIdx) * inputs2D.get(batchRowIdx, inIdx);
                }
                outputs2D.add(weightedInputSum, batchRowIdx, outIdx); // add weighted sum of inputs  - mora add a ne set  zbog biasa!!!!!! ili gore u weighted sum dodaj bias a onda ovde mozes set
            }
        }
    }

    /**
     * Performs backward pass for this layer.
     */
    @Override
    public void backward() {

        if (!isTrainable()) {
            return;
        }

        deltas.copyFrom(outputErrors); // ovo bi bilo sjajno da moze da gleda isti kao da bude view - instanca koja sharuje memory space
        if (batchMode) deltas.div(batchSize);
        
        // todo add cuda backward
        if (DeepNetts.getInstance().useCuda()) {
            deltas.copyToGPU();
            backwardImpl.backward();
            return;
        }
  
        backwardImpl.backward();

        // ovo je softmax za u kombinaciji sa cross entropy loss funkcijom, nisam siguran da li je dobra u kombinaciji sa sigmoidnom? Mada tada ide BinaryCE i obican output layer
//        for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) { // iterate all output neurons / deltas
//            // da li delta treba da s emnozi sa izvodom? izgleda da ne treba
//            // prema ovome ne mora file:///D:/DeepNettsSredjivanje/Books%20and%20Tuts/PetersNotes%20CE%20Softmax%20Derivation/Peter's%20NotesCE.html
//            // http://neuralnetworksanddeeplearning.com/chap3.html
//            for (int inCol = 0; inCol < inputs.getCols(); inCol++) { // prev layer is allways Dense. iterate all inputs/weights for the current neuron
//               final float regularization = (regL2 !=0 ?
//                                                 regL2 * 2 * weights.get(inCol, deltaCol) :
//                                                 regL1 * MathFunctions.absPrime(weights.get(inCol, deltaCol)));                 
//                
//                final float grad = deltas.get(deltaCol) * inputs.get(inCol) + regularization; 
//                gradients.set(grad, inCol, deltaCol);
//                final float deltaWeight = optimizer.calculateDeltaWeight(grad, inCol, deltaCol);   
//                deltaWeights.add(deltaWeight, inCol, deltaCol);
//            }
//            
//            final float deltaBias = optimizer.calculateDeltaBias(deltas.get(deltaCol), deltaCol); 
//            deltaBiases[deltaCol] += deltaBias;
//        }
    }

    private class SingleThreadedBackward implements Backward {

        @Override
        public void backward() {
            deltaWeights.fill(0);
            deltaBiases.fill(0);

            int from = 0;
            int to = deltas.numElements();

            Tensor1D inputs1D = (Tensor1D) prevLayer.getOutputs();
            Tensor1D deltas1D = (Tensor1D) deltas;
            Tensor2D weights2D = (Tensor2D) weights;
            Tensor2D gradients2D = (Tensor2D) gradients;

            for (int deltaIdx = from; deltaIdx < to; deltaIdx++) { // iterate all output neurons / deltas
                // da li delta treba da s emnozi sa izvodom? izgleda da ne treba
                // prema ovome ne mora file:///D:/DeepNettsSredjivanje/Books%20and%20Tuts/PetersNotes%20CE%20Softmax%20Derivation/Peter's%20NotesCE.html
                // http://neuralnetworksanddeeplearning.com/chap3.html
                for (int inIdx = 0; inIdx < inputs.numElements(); inIdx++) { // prev layer is allways Dense. iterate all inputs/weights for the current neuron
                    final float regularization = (regL2 != 0
                            ? regL2 * 2 * weights2D.get(deltaIdx, inIdx)
                            : regL1 * MathFunctions.absPrime(weights2D.get(deltaIdx, inIdx)));

                    final float grad = deltas1D.get(deltaIdx) * inputs1D.get(inIdx) + regularization;
                    gradients2D.set(grad, deltaIdx, inIdx); // ovo moze da se ukljucuje za potrebe debugovanja - not used anywhere
                    final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
                    deltaWeights.add(deltaWeight, deltaIdx, inIdx);
                }

                // ovo treba da bude ukljuceno!!!
                final float deltaBias = optimizer.calculateDeltaBias(deltas1D.get(deltaIdx), deltaIdx);
                deltaBiases.add(deltaBias, deltaIdx);
            }
        }

    }

    private class SingleThreadedBackwardBatch implements Backward {

        @Override
        public void backward() {
            Tensor2D inputs2D = (Tensor2D) prevLayer.getOutputs();
            Tensor2D deltas2D = (Tensor2D) deltas;
            Tensor2D weights2D = (Tensor2D) weights;
            Tensor2D gradients2D = (Tensor2D) gradients;

            //      prva implementacija single threaded ubaci da izvrti batch u for petlji, onda paralelizuj batch 
        //    deltas2D.div(batchSize);
            for (int batchIdx = 0; batchIdx < inputs2D.cols(); batchIdx++) {
                for (int deltaIdx = 0; deltaIdx < deltas2D.rows(); deltaIdx++) { // iterate all output neurons / deltas
                    // da li delta treba da s emnozi sa izvodom? izgleda da ne treba
                    // prema ovome ne mora file:///D:/DeepNettsSredjivanje/Books%20and%20Tuts/PetersNotes%20CE%20Softmax%20Derivation/Peter's%20NotesCE.html
                    // http://neuralnetworksanddeeplearning.com/chap3.html
                    for (int inIdx = 0; inIdx < inputs2D.rows(); inIdx++) { // prev layer is allways Dense. iterate all inputs/weights for the current neuron
                        final float regularization = (regL2 != 0
                                ? regL2 * 2 * weights2D.get(deltaIdx, inIdx)
                                : regL1 * MathFunctions.absPrime(weights2D.get(deltaIdx, inIdx)));

                        final float grad = deltas2D.get(deltaIdx, batchIdx) * inputs2D.get(inIdx, batchIdx) + regularization;
                        gradients2D.add(grad, deltaIdx, inIdx); // ovo moze da se ukljucuje za potrebe debugovanja - not used anywhere
                        final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
                        deltaWeights.add(deltaWeight, deltaIdx, inIdx);
                    }

                    // ovo treba da bude ukljuceno!!!
                    final float deltaBias = optimizer.calculateDeltaBias(deltas2D.get(deltaIdx, batchIdx), deltaIdx);
                    deltaBiases.add(deltaBias, deltaIdx);
                }
            }
        }

    }

    private class MultiThreadedBackward implements Backward {

        @Override
        public void backward() {
            try {
                threadPool.run(backwardTasks);
            } catch (InterruptedException ex) {
                //LOG.warning(ex.getMessage());
            }
        }

    }

    private void backwardForCellRange(int from, int to) {
        Tensor1D inputs1D = (Tensor1D) prevLayer.getOutputs();
        Tensor1D deltas1D = (Tensor1D) deltas;
        Tensor2D weights2D = (Tensor2D) weights;
        Tensor2D gradients2D = (Tensor2D) gradients;

        for (int deltaIdx = from; deltaIdx < to; deltaIdx++) { // iterate all output neurons / deltas
            // da li delta treba da s emnozi sa izvodom? izgleda da ne treba
            // prema ovome ne mora file:///D:/DeepNettsSredjivanje/Books%20and%20Tuts/PetersNotes%20CE%20Softmax%20Derivation/Peter's%20NotesCE.html
            // http://neuralnetworksanddeeplearning.com/chap3.html
            for (int inIdx = 0; inIdx < inputs.numElements(); inIdx++) { // prev layer is allways Dense. iterate all inputs/weights for the current neuron
                final float regularization = (regL2 != 0
                        ? regL2 * 2 * weights2D.get(deltaIdx, inIdx)
                        : regL1 * MathFunctions.absPrime(weights2D.get(deltaIdx, inIdx)));

                final float grad = deltas1D.get(deltaIdx) * inputs1D.get(inIdx) + regularization;
                gradients2D.set(grad, deltaIdx, inIdx); // ovo moze da se ukljucuje za potrebe debugovanja - not used anywhere
                final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);
                deltaWeights.add(deltaWeight, deltaIdx, inIdx);
            }

            // ovo treba da bude ukljuceno!!!
            final float deltaBias = optimizer.calculateDeltaBias(deltas1D.get(deltaIdx), deltaIdx);
            deltaBiases.add(deltaBias, deltaIdx);
        }
    }

//    private void backwardBatch() {
//         Tensor2D inputs2D = (Tensor2D)prevLayer.getOutputs(); // kako je ovaj 1d???
//         Tensor2D deltas2D = (Tensor2D)deltas;        
//         Tensor2D weights2D = (Tensor2D)weights;
//         Tensor2D gradients2D = (Tensor2D)gradients;                 
//         //deltas2D = weights.dotProd(errorsTransposed?, deltas2D); // ovo ce da zameni ovo ispod         
//
    ////      prva implementacija single threaded ubaci da izvrti batch u for petlji, onda paralelizuj batch 
//
//        deltas2D.div(batchSize);
//        for(int batchIdx = 0; batchIdx < inputs2D.cols(); batchIdx++) {    
//            for (int deltaIdx = 0; deltaIdx < deltas2D.rows(); deltaIdx++) { // iterate all output neurons / deltas
//                // da li delta treba da s emnozi sa izvodom? izgleda da ne treba
//                // prema ovome ne mora file:///D:/DeepNettsSredjivanje/Books%20and%20Tuts/PetersNotes%20CE%20Softmax%20Derivation/Peter's%20NotesCE.html
//                // http://neuralnetworksanddeeplearning.com/chap3.html
//                for (int inIdx = 0; inIdx < inputs2D.rows(); inIdx++) { // prev layer is allways Dense. iterate all inputs/weights for the current neuron
//                   final float regularization = (regL2 !=0 ?
//                                                 regL2 * 2 * weights2D.get(deltaIdx, inIdx) :
//                                                 regL1 * MathFunctions.absPrime(weights2D.get(deltaIdx, inIdx)));                 
//
//                    final float grad = deltas2D.get(deltaIdx, batchIdx) * inputs2D.get(inIdx, batchIdx) + regularization;
//                    gradients2D.add(grad, deltaIdx, inIdx); // ovo moze da se ukljucuje za potrebe debugovanja - not used anywhere
//                    final float deltaWeight = optimizer.calculateDeltaWeight(grad, deltaIdx, inIdx);   
//                    deltaWeights.add(deltaWeight, deltaIdx, inIdx);
//                }
//
//                // ovo treba da bude ukljuceno!!!
//                final float deltaBias = optimizer.calculateDeltaBias(deltas2D.get(deltaIdx, batchIdx), deltaIdx); 
//                deltaBiases.add(deltaBias, deltaIdx);
//            }    
//        }
//        
//    }
      
    public void applySoftmax(Tensor1D outputs1D, float maxWs) {

        if (DeepNetts.getInstance().useVectorAPI()) {
            //applySoftmaxVectorized(outputs1D, maxWs);
            vectorizationImpl.applySoftmaxVectorized(outputs1D, maxWs);            
            return;
        }

        // find max weighted sum
        for (int outIdx = 0; outIdx < outputs1D.numElements(); outIdx++) {                    // for all neurons in this layer
            if (outputs1D.get(outIdx) > maxWs) {
                maxWs = outputs1D.get(outIdx);
            }
        }

        // calculate outputs and denominator sum (use max for numerical stability)
        float denSum = 0;
        for (int outIdx = 0; outIdx < outputs1D.numElements(); outIdx++) {
            outputs1D.set((float) Math.exp(outputs1D.get(outIdx) - maxWs), outIdx); // maxWs used for numerical stability
            denSum += outputs1D.get(outIdx);
        }

        outputs1D.div(denSum); // scale all outputs to sum to 1

    }

    public void applySoftmaxBatch(Tensor2D inputs2D, Tensor2D outputs2D) {

        if (DeepNetts.getInstance().useVectorAPI()) {
            //applySoftmaxBatchVectorized(outputs2D);
            vectorizationImpl.applySoftmaxBatchVectorized(outputs2D);
            return;
        }

        for (int batchIdx = 0; batchIdx < inputs2D.cols(); batchIdx++) {
            float maxWs = Float.NEGATIVE_INFINITY;
            for (int outIdx = 0; outIdx < outputs2D.rows(); outIdx++) {                    // for all neurons in this layer
                if (outputs2D.get(outIdx, batchIdx) > maxWs) {
                    maxWs = outputs2D.get(outIdx, batchIdx);
                }
            }

            // calculate outputs and denominator sum (use max for numerical stability)
            float denSum = 0;
            for (int outIdx = 0; outIdx < outputs2D.rows(); outIdx++) {
                outputs2D.set((float) Math.exp(outputs2D.get(outIdx, batchIdx) - maxWs), outIdx, batchIdx); // maxWs used for numerical stability
                denSum += outputs2D.get(outIdx, batchIdx);
            }

            for (int outIdx = 0; outIdx < outputs2D.rows(); outIdx++) {
                final float newVal = outputs2D.get(outIdx, batchIdx) / denSum;
                outputs2D.set(newVal, outIdx, batchIdx);
            }
            //outputs.div(denSum); // scale all outputs to sum to 1             
        }
    }

//    private void applySoftmaxVectorized(Tensor1D outputs1D, float maxWs) {
//        int upperBound = SPECIES.loopBound(outputs1D.numElements());
//
//        // 1.  nadji maksimalnu vrednost (zbog numericke stabilnosti)
//        int i = 0;
//        for (; i < upperBound; i += vecLen) {
//            FloatVector vec = FloatVector.fromArray(SPECIES, outputs1D.getValues(), i);
//            float laneMax = vec.reduceLanes(VectorOperators.MAX);
//            maxWs = Math.max(maxWs, laneMax);
//        }
//        for (; i < outputs1D.numElements(); i++) {
//            maxWs = Math.max(maxWs, outputs1D.get(i));
//        }
//
//        // 2. Oduzmi maxWs i izracunaj e^(x - maxWs)
//        FloatVector maxWsVec = FloatVector.broadcast(SPECIES, maxWs);
//        i = 0;
//        for (; i < upperBound; i += vecLen) {
//            FloatVector vec = FloatVector.fromArray(SPECIES, outputs1D.getValues(), i);
//            FloatVector vecSub = vec.sub(maxWsVec);                        // x - maxWs
//            FloatVector expVec = vecSub.lanewise(VectorOperators.EXP);    // e^(x - maxWs)
//            expVec.intoArray(outputs1D.getValues(), i);
//        }
//        for (; i < outputs1D.numElements(); i++) {
//            outputs1D.set(((float) Math.exp(outputs1D.get(i) - maxWs)), i);  // Ispravljeno!
//        }
//
//        // 3. Sabiramo sve e^(x - maxWs) vrednosti
//        FloatVector sumVec = FloatVector.zero(SPECIES);
//        i = 0;
//        for (; i < upperBound; i += vecLen) {
//            FloatVector vec = FloatVector.fromArray(SPECIES, outputs1D.getValues(), i);
//            sumVec = sumVec.add(vec);
//        }
//        float denominatorSum = sumVec.reduceLanes(VectorOperators.ADD);
//        for (; i < outputs1D.numElements(); i++) {
//            denominatorSum += outputs1D.get(i);
//        }
//
//        // 4. Normalizuj: softmax = e^(x - maxWs) / suma
//        FloatVector denSumVec = FloatVector.broadcast(SPECIES, denominatorSum);
//        i = 0;
//        for (; i < upperBound; i += vecLen) {
//            FloatVector vec = FloatVector.fromArray(SPECIES, outputs1D.getValues(), i);
//            FloatVector normalized = vec.div(denSumVec);
//            normalized.intoArray(outputs1D.getValues(), i);
//        }
//        for (; i < outputs1D.numElements(); i++) {
//            outputs1D.set(outputs1D.get(i) / denominatorSum, i);
//        }
//
//    }
//
//    public void applySoftmaxBatchVectorized(Tensor2D logits) {
//        int C = logits.rows();  // broj klasa (redovi)
//        int B = logits.cols();  // batch size (kolone)
//
//        float[] values = logits.getValues(); // column-major: (c, b) → b * C + c
//        int upperBound = SPECIES.loopBound(C);
//
//        for (int b = 0; b < B; b++) {
//            int base = b * C;
//
//            // === 1. Nadji max vrednost u koloni radi numericke stabilnosti ===
//            float maxWs = Float.NEGATIVE_INFINITY;
//            int i = 0;
//            for (; i < upperBound; i += vecLen) {
//                FloatVector vec = FloatVector.fromArray(SPECIES, values, base + i);
//                float laneMax = vec.reduceLanes(VectorOperators.MAX);
//                maxWs = Math.max(maxWs, laneMax);
//            }
//            for (; i < C; i++) {
//                maxWs = Math.max(maxWs, values[base + i]);
//            }
//
//            // === 2. Oduzmi maxWs i izracunaj e^(x - maxWs) ===
//            FloatVector maxVec = FloatVector.broadcast(SPECIES, maxWs);
//            i = 0;
//            for (; i < upperBound; i += vecLen) {
//                FloatVector vec = FloatVector.fromArray(SPECIES, values, base + i);
//                FloatVector expVec = vec.sub(maxVec).lanewise(VectorOperators.EXP);
//                expVec.intoArray(values, base + i);
//            }
//            for (; i < C; i++) {
//                values[base + i] = (float) Math.exp(values[base + i] - maxWs);
//            }
//
//            // === 3. Racuanj sumu e^(x - maxWs) ===
//            FloatVector sumVec = FloatVector.zero(SPECIES);
//            i = 0;
//            for (; i < upperBound; i += vecLen) {
//                FloatVector vec = FloatVector.fromArray(SPECIES, values, base + i);
//                sumVec = sumVec.add(vec);
//            }
//            float denomSum = sumVec.reduceLanes(VectorOperators.ADD);
//            for (; i < C; i++) {
//                denomSum += values[base + i];
//            }
//
//            // === 4. Podeli svaki sa sumom da se dobije softmax ===
//            FloatVector denomVec = FloatVector.broadcast(SPECIES, denomSum);
//            i = 0;
//            for (; i < upperBound; i += vecLen) {
//                FloatVector vec = FloatVector.fromArray(SPECIES, values, base + i);
//                FloatVector normVec = vec.div(denomVec);
//                normVec.intoArray(values, base + i);
//            }
//            for (; i < C; i++) {
//                values[base + i] = values[base + i] / denomSum;
//            }
//        }
//    }

}
