package deepnetts.net.layers;

import deepnetts.accl.spi.AcceleratorService;
import deepnetts.core.DeepNetts;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;
import deepnetts.util.CallableRangeConsumer;
import deepnetts.util.DeepNettsThreadPool;
import java.util.ArrayList;
import java.util.concurrent.Callable;

/**
 * Transforms outputs from previous 3D layer into a flatten 1D tensor in forward
 * pass, Backward pass propagates weighted errors/deltas from the next fully
 * connected layer. Automatically added after 2D or 3D layer to transition to
 * fully connected layers.
 */
public class FlattenLayer extends AbstractLayer<TensorBase, TensorBase, TensorBase> {

    private transient boolean multithreaded = false;

    public FlattenLayer() {
        super(ActivationType.LINEAR);
    }

    @Override
    public void init() {
        inputs = prevLayer.getOutputs();

        if (inputs instanceof Tensor3D) { // non batch mode
            int prevSize = prevLayer.getOutputs().numElements();
            batchSize = 1;
            this.width = prevSize;
            this.outputs = new Tensor1D(prevSize);
            this.deltas = new Tensor1D(prevSize);
        } else if (inputs instanceof Tensor4D) { // batch mode
            batchMode = true;
            batchSize = ((Tensor4D) inputs).fourthDim(); 
            int prevSize = prevLayer.getOutputs().numElements() / batchSize;
            this.width = prevSize;
            this.outputs = new Tensor2D(prevSize, batchSize);
            this.deltas = new Tensor2D(prevSize, batchSize);
        }

        initTransientFields();
    }
    
    @Override
    public void initTransientFields() {
        numThreads = DeepNetts.getInstance().getMaxThreads();
         multithreaded = (numThreads > 1);
         
        if (DeepNetts.getInstance().useCuda()) {
        //    forward = new FlattenForwardCuda(cudaHandles, this);
            forwardImpl = AcceleratorService.defaultProvider().createFlattenForwardAcc(cudaHandles, this);// new FlattenForwardCuda(cudaHandles, this);
          //  backward = new FlattenBackwardCuda(cudaHandles, this);        
            backwardImpl = AcceleratorService.defaultProvider().createFlattenBackwardAcc(cudaHandles, this); // new FlattenBackwardCuda(cudaHandles, this);        
        } else if (!multithreaded) { // single threaded
            if (!batchMode) { // inputs instanceof Tensor1D
                forwardImpl = new SingleThreadedForward(); // default forward
                backwardImpl = new SingleThreadedBackward();
            } else if (batchMode) {
                forwardImpl = new SingleThreadedForwardBatch();
                backwardImpl = new SingleThreadedBackwardBatch();
            }
        } else { // multithreaded
                forwardImpl = new MultiThreadedForward();
                backwardImpl = new SingleThreadedBackward(); 
            // TODO: batch varijantu
//            if (!batchMode) { 
//                forward = new MultiThreadedForward();
//                backward = new MultiThreadedBackward(); 
//            } else {
//                throw new RuntimeException("Not implemented");
//            }
        }
    }    

    @Override
    public void forward() {
        // just copy all outputs from the previous layer to tensor with flatten shape
    /*        
    for(r : rows) # 0..4
        for(c : cols)   #  0..6
            for(ch : depth) # 0..2
                idx = r * cols * depth + c * depth + ch
                flatten[idx] = tensor[0, r, c, ch]   // ovo je layout      
             */

        forwardImpl.forward();

    }

    @Override
    public void backward() {
        // prenesi weighted deltas iz sledeceg fc layer-a
        // ovo isto da se paralelizuje
        backwardImpl.backward();
    }

    @Override
    public void setOptimizerType(OptimizerType optType) {
        // does nothing here
    }

    @Override
    public void applyWeightChanges() {
        // does nothing - no weights
    }

    @Override
    public String toString() {
        return "Flatten Layer { width:" + width + " }";
    }

    private class SingleThreadedForward implements Forward {

        @Override
        public void forward() {
            Tensor3D prevOut = (Tensor3D) prevLayer.getOutputs();
            final int prevOutCols = prevOut.cols();
            final int prevOutDepth = prevOut.depth();
            Tensor1D outputs1D = (Tensor1D) outputs;

            for (int row = 0; row < prevOut.rows(); row++) {
                for (int col = 0; col < prevOut.cols(); col++) {
                    for (int ch = 0; ch < prevOut.depth(); ch++) {
                        final int idx = row * prevOutCols * prevOutDepth + col * prevOutDepth + ch;
                        outputs1D.set(prevOut.get(ch, row, col), idx);
                    }
                }
            }
        }
    }
    
    private class MultiThreadedForward implements Forward {
        private final transient ArrayList<Callable<Void>> forwardTasks;
        private final Tensor3D prevOut;
        private final Tensor1D outputs1D;

        public MultiThreadedForward() {          
          prevOut = (Tensor3D) prevLayer.getOutputs();
          outputs1D = (Tensor1D) outputs;               
          
          forwardTasks = new ArrayList<>();
          int[] cellsPerThread = DeepNettsThreadPool.calculateCellsPerThread(width, numThreads);
          int from = 0, to = 0;
          

                for (int i = 0; i < numThreads; i++) {
                    from = to;
                    to = from + cellsPerThread[i];

                    CallableRangeConsumer ftask = new CallableRangeConsumer(from, to, this::forwardForRange); // forwardForCellRange
                    forwardTasks.add(ftask);
                }                                
        }

        @Override
        public void forward() {
            try {
                threadPool.run(forwardTasks);
            } catch (InterruptedException ex) {
             //   LOG.warning(ex.getMessage());
            }
        }
        
        private void forwardForRange(final int from, final int to) {
            final int prevOutCols = prevOut.cols();
            final int prevOutDepth = prevOut.depth();
            
            //for (int row = 0; row < prevOut.rows(); row++) {
            for (int row = from; row < to; row++) {
                for (int col = 0; col < prevOut.cols(); col++) {
                    for (int ch = 0; ch < prevOut.depth(); ch++) {
                        final int idx = row * prevOutCols * prevOutDepth + col * prevOutDepth + ch;
                        outputs1D.set(prevOut.get(ch, row, col), idx);
                    }
                }
            }            
        }
    }   
    
    // za ovo ni nemamo test ali izgenerisi od single threaded backwarda od flattena neki mnist ili sl
    private class MultiThreadedBackward implements Backward {
        private final transient ArrayList<Callable<Void>> backwardTasks;

        public MultiThreadedBackward() {
            this.backwardTasks =  new ArrayList<>();
            
          int[] cellsPerThread = DeepNettsThreadPool.calculateCellsPerThread(width, numThreads);
          int from = 0, to = 0;
          

                for (int i = 0; i < numThreads; i++) {
                    from = to;
                    to = from + cellsPerThread[i];

                    CallableRangeConsumer btask = new CallableRangeConsumer(from, to, this::backwardForRange); // forwardForCellRange
                    backwardTasks.add(btask);
                }               
        }
        
        
        
        @Override
        public void backward() {
            deltas.fill(0);
            try {
                threadPool.run(backwardTasks);
            } catch (InterruptedException ex) {
               // LOG.warning(ex.getMessage());
            }
        }

        private void backwardForRange(int from, int to) {
             final TensorBase nextLayerWeights = nextLayer.weights;
            final Tensor2D nextLayerWeights2D = (Tensor2D) nextLayerWeights;

            final Tensor1D nextLayerDeltas = (Tensor1D) nextLayer.deltas;
            //for (int deltaIdx = 0; deltaIdx < deltas.numElements(); deltaIdx++) {   // for every unit/delta in this layer
            for (int deltaIdx = from; deltaIdx < to; deltaIdx++) {   // for every unit/delta in this layer
                for (int ndIdx = 0; ndIdx < nextLayerDeltas.numElements(); ndIdx++) { // iterate all deltas from next layer
                    final float weightedDelta = nextLayerDeltas.get(ndIdx) * nextLayerWeights2D.get(ndIdx, deltaIdx); // propagate weighted deltas from next layer
                    ((Tensor1D) deltas).add(weightedDelta, deltaIdx); // calculate weighted sum of deltas from the next layer
                }
            } // end sum weighted deltas from next layer       
        }
    }    

    private class SingleThreadedBackward implements Backward {

        @Override
        public void backward() {
            final TensorBase nextLayerWeights = nextLayer.weights;
            final Tensor2D nextLayerWeights2D = (Tensor2D) nextLayerWeights;

            final Tensor1D nextLayerDeltas = (Tensor1D) nextLayer.deltas;
            deltas.fill(0);
            for (int deltaIdx = 0; deltaIdx < deltas.numElements(); deltaIdx++) {   // for every unit/delta in this layer
                for (int ndIdx = 0; ndIdx < nextLayerDeltas.numElements(); ndIdx++) { // iterate all deltas from next layer
                    final float weightedDelta = nextLayerDeltas.get(ndIdx) * nextLayerWeights2D.get(ndIdx, deltaIdx); // propagate weighted deltas from next layer
                    ((Tensor1D) deltas).add(weightedDelta, deltaIdx); // calculate weighted sum of deltas from the next layer
                }
            } // end sum weighted deltas from next layer       
        }
    }

    private class SingleThreadedForwardBatch implements Forward {

        @Override
        public void forward() {
            Tensor4D prevOut = (Tensor4D) prevLayer.getOutputs();
            final int prevOutCols = prevOut.cols();
            final int prevOutDepth = prevOut.depth();

            // batch mode impl here - isto kao i ovo gore samo ubaci jos jedan for za batch
            Tensor2D outputs2D = (Tensor2D) outputs;
            for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
                for (int row = 0; row < prevOut.rows(); row++) {
                    for (int col = 0; col < prevOut.cols(); col++) {
                        for (int ch = 0; ch < prevOut.depth(); ch++) {
                            final int idx = row * prevOutCols * prevOutDepth + col * prevOutDepth + ch;
                            outputs2D.set(prevOut.get(batchIdx, ch, row, col), idx, batchIdx);
                        }
                    }
                }
            }
        }
    }

    private class SingleThreadedBackwardBatch implements Backward {

        @Override
        public void backward() {
            final TensorBase nextLayerWeights = nextLayer.weights;
            final Tensor2D nextLayerWeights2D = (Tensor2D) nextLayerWeights;

            final Tensor2D nextLayerDeltas = (Tensor2D) nextLayer.deltas;
            final Tensor2D deltas2d = (Tensor2D) deltas;

            deltas.fill(0);
            for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
                for (int deltaIdx = 0; deltaIdx < deltas2d.rows(); deltaIdx++) {   // for every unit/delta in this layer
                    for (int ndIdx = 0; ndIdx < nextLayerDeltas.rows(); ndIdx++) { // iterate all deltas from next layer
                        final float weightedDelta = nextLayerDeltas.get(ndIdx, batchIdx) * nextLayerWeights2D.get(ndIdx, deltaIdx); // propagate weighted deltas from next layer
                        deltas2d.add(weightedDelta, deltaIdx, batchIdx); // calculate weighted sum of deltas from the next layer
                    }
                } // end sum weighted deltas from next layer         
            }
        }
    }
}
