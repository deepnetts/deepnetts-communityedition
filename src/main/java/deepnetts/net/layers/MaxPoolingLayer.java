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

import deepnetts.accl.spi.AcceleratorService;
import deepnetts.core.DeepNetts;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.logging.Logger;

/**
 * This layer performs max pooling operation in convolutional neural network,
 * which scales down output from previous layer by taking max outputs from small
 * predefined filter areas.
 *
 * @see ConvolutionalNetwork
 * @author Zoran Sevarac
 */
public final class MaxPoolingLayer extends AbstractLayer<TensorBase, TensorBase, TensorBase> {

    // I i O mogu  biti Tenso3D ili Tensor4D i spreman je za gpu
    private static final long serialVersionUID = 6257978737942468865L;

    /**
     * Filter dimensions.
     *
     * Commonly used 2x2 with stride 2
     */
    final int filterWidth, filterHeight;

    /**
     * Filter step.
     *
     * Commonly used 2
     */
    final int stride;

    /**
     * Max activation idxs.
     *
     * Remember idx of max output for each filter position.
     * [channel][row][col][2]
     */
    int maxIdx[][][][];
    int maxIdxBatch[][][][][];

    private transient boolean multithreaded = false;
    private transient List<Callable<Void>> forwardTasks;
    private transient List<Callable<Void>> backwardTasks;
    private transient List<Callable<Void>> backwardConvTasks;


    private static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());

    /**
     * Creates a new max pooling layer with specified filter dimensions and
     * stride.
     *
     * @param filterWidth width of the filter square
     * @param filterHeight height of the filter square
     * @param stride filter step
     */
    public MaxPoolingLayer(int filterWidth, int filterHeight, int stride) {
        super(ActivationType.LINEAR);
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.stride = stride;
    }

    public MaxPoolingLayer(Filter filter) {
        super(ActivationType.LINEAR);
        this.filterWidth = filter.getWidth();
        this.filterHeight = filter.getHeight();
        this.stride = filter.getStride();
    }

    @Override
    final public void init() {
        // max pooling layer can be only after Convolutional Layer
        if (!(prevLayer instanceof ConvolutionalLayer)) {
            throw new RuntimeException("Illegal network architecture! MaxPooling can be only after convolutional layer!");
        }

        inputs = prevLayer.getOutputs();

        if (inputs instanceof Tensor3D) {
            Tensor3D inputs3d = (Tensor3D) prevLayer.getOutputs();

            // proveri da li je ovo dobro da li treba cols ili rows, zavisi kako pakujes slike i layere
            width = (inputs3d.cols() - filterWidth) / stride + 1; // ovo mora biti ceo broj strude veci od 2, 3 je suvise destruktivan
            height = (inputs3d.rows() - filterHeight) / stride + 1;
            depth = inputs3d.depth(); // depth of pooling layer is always same as in previous convolutional layer

            outputs = new Tensor3D(depth, height, width);
            deltas = new Tensor3D(depth, height, width);

            // used in fprop to save idx position of max value
            maxIdx = new int[depth][height][width][2]; // svakoj poziciji filtera odgovara jedna [row, col] celija u outputu idx 0 je col, idx 1 je row
        } else if (inputs instanceof Tensor4D) {
            Tensor4D inputs4d = (Tensor4D) prevLayer.getOutputs();
            batchMode = true;
            batchSize = inputs4d.fourthDim();

            // proveri da li je ovo dobro da li treba cols ili rows, zavisi kako pakujes slike i layere
            width = (inputs4d.cols() - filterWidth) / stride + 1; // ovo mora biti ceo broj strude veci od 2, 3 je suvise destruktivan
            height = (inputs4d.rows() - filterHeight) / stride + 1;
            depth = inputs4d.depth(); // depth of pooling layer is always same as in previous convolutional layer

            outputs = new Tensor4D(batchSize, depth, height, width);
            deltas = new Tensor4D(batchSize, depth, height, width);

            // used in fprop to save idx position of max value
            // e ovo bih morao da menjam i umesto niza da isto stavim tenzor za batch - ali nemam 5D
            maxIdxBatch = new int[batchSize][depth][height][width][2]; // svakoj poziciji filtera odgovara jedna [row, col] celija u outputu idx 0 je col, idx 1 je row            
        }

        initTransientFields();
    }

    @Override
    public void initTransientFields() {
        //int threadCount = Math.min(DeepNetts.getInstance().getMaxThreads(), depth);
        int threadCount = DeepNetts.getInstance().getMaxThreads();
        if (threadCount > 1) {
            multithreaded = true;
            int[] channelsPerThread = calculateChannelsPerThread(threadCount);

            forwardTasks = new ArrayList<>();
            backwardTasks = new ArrayList<>();
            backwardConvTasks = new ArrayList<>();

            int fromCh = 0, toCh = 0;
            for (int i = 0; i < threadCount; i++) {
                fromCh = toCh;
                toCh = fromCh + channelsPerThread[i];
                // ovde koristi CallableRangeConsumer i za backward isto, proveri da li isto rade i potvrdi
                ForwardCallable task = new ForwardCallable(fromCh, toCh);
                forwardTasks.add(task);

                if (nextLayer instanceof FlattenLayer) {
                    BackwardFromFlattenCallable bfctask = new BackwardFromFlattenCallable(fromCh, toCh);
                    backwardTasks.add(bfctask);
                } else if (nextLayer instanceof ConvolutionalLayer) {
                    BackwardFromConvolutionalCallable bctask = new BackwardFromConvolutionalCallable(fromCh, toCh);
                    backwardConvTasks.add(bctask);
                }
            }
        }

        if (DeepNetts.getInstance().useCuda()) {
          //  forward = new MaxPoolingForwardCuda(cudaHandles, this);
            forwardImpl = AcceleratorService.defaultProvider().createMaxpoolingForwardAcc(cudaHandles, this);// new MaxPoolingForwardCuda(cudaHandles, this);
           // backward = new MaxPoolingBackwardCuda(cudaHandles, this);
            backwardImpl = AcceleratorService.defaultProvider().createMaxpoolingBackwardAcc(cudaHandles, this);
        } else if (!multithreaded) { // single threaded
            if (!batchMode) { // inputs instanceof Tensor1D
                forwardImpl = new SingleThreadedForward(); // default forward
            } else if (batchMode) {
                forwardImpl = new SingleThreadedForwardBatch();
            }
        } else { // multithreaded
            forwardImpl = new MultiThreadedForward();
        }
    }

    /**
     * Max pooling forward pass outputs the max value for each filter position.
     */
    @Override
    public void forward() {
    //    inputs = prevLayer.getOutputs(); // ako slucajno treba zbog cuda-e

        forwardImpl.forward();        
    }

    private class SingleThreadedForward implements Forward {

        @Override
        public void forward() {
            int fromCh = 0, toCh = depth;

            Tensor3D inputs3D = (Tensor3D) inputs;
            Tensor3D outputs3D = (Tensor3D) outputs;

            for (int ch = fromCh; ch < toCh; ch++) {
                float max; // max value
                int maxC = -1, maxR = -1;

                int outCol = 0, outRow = 0;

                for (int inRow = 0; inRow < inputs3D.rows() - filterHeight + 1; inRow += stride) {
                    outCol = 0; // reset col on every new row
                    for (int inCol = 0; inCol < inputs3D.cols() - filterWidth + 1; inCol += stride) {

                        // apply max pool filter
                        max = inputs3D.get(ch, inRow, inCol);
                        maxC = inCol;
                        maxR = inRow;
                        for (int fc = 0; fc < filterWidth; fc++) {    // this order in orfder to conform to gpu impl                        
                            for (int fr = 0; fr < filterHeight; fr++) {
                        
                                final float inputVal = inputs3D.get(ch, inRow + fr, inCol + fc);
                                if (inputVal > max) {
                                    max = inputVal;
                                    maxR = inRow + fr;
                                    maxC = inCol + fc;
                                }
                            }
                        }

                        // zapamti indexe neurona iz prethodnog lejera koji su propustili max (koristice se u bacward pass-u)
                        maxIdx[ch][outRow][outCol][0] = maxR; // height idx (row)
                        maxIdx[ch][outRow][outCol][1] = maxC; // width idx (col)

                        outputs3D.set(max, ch, outRow, outCol); // set max value as output
                        outCol++;   // increase output col by one for each input (stride) step
                    } // scan col
                    outRow++; // increase output row by one for each input (stride) step
                } // scan row
            }
        }
    }

    private class SingleThreadedForwardBatch implements Forward {

        @Override
        public void forward() {
            for (int ch = 0; ch < depth; ch++) {
                forwardForChannelBatch(ch);
            }
        }

    }

    private class MultiThreadedForward implements Forward {

        @Override
        public void forward() {
            try {
                threadPool.run(forwardTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage());
            }
        }
    }

    private void forwardForChannelRange(final int fromCh, final int toCh) {
        Tensor3D inputs3D = (Tensor3D) inputs;
        Tensor3D outputs3D = (Tensor3D) outputs;

        for (int ch = fromCh; ch < toCh; ch++) {
            float max; // max value
            int maxC = -1, maxR = -1;

            int outCol = 0, outRow = 0;

            for (int inRow = 0; inRow < inputs3D.rows() - filterHeight + 1; inRow += stride) {
                outCol = 0; // reset col on every new row
                for (int inCol = 0; inCol < inputs3D.cols() - filterWidth + 1; inCol += stride) {

                    // apply max pool filter
                    max = inputs3D.get(ch, inRow, inCol);
                    maxC = inCol;
                    maxR = inRow;
                    for (int fr = 0; fr < filterHeight; fr++) {
                        for (int fc = 0; fc < filterWidth; fc++) {
                            final float inputVal = inputs3D.get(ch, inRow + fr, inCol + fc);
                            if (inputVal >= max) {
                                max = inputVal;
                                maxR = inRow + fr;
                                maxC = inCol + fc;
                            }
                        }
                    }

                    // zapamti indexe neurona iz prethodnog lejera koji su propustili max (koristice se u bacward pass-u)
                    maxIdx[ch][outRow][outCol][0] = maxR; // height idx (row)
                    maxIdx[ch][outRow][outCol][1] = maxC; // width idx (col)

                    outputs3D.set(max, ch, outRow, outCol); // set max value as output
                    outCol++;   // increase output col by one for each input (stride) step
                } // scan col
                outRow++; // increase output row by one for each input (stride) step
            } // scan row
        }
    }

    private void forwardForChannelBatch(final int ch) {
        final Tensor4D inputs4D = (Tensor4D) inputs;
        final Tensor4D outputs4D = (Tensor4D) outputs;

        float max; // max value
        int maxC = -1, maxR = -1;

        for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            int outCol = 0, outRow = 0;

            for (int inRow = 0; inRow < inputs4D.rows() - filterHeight + 1; inRow += stride) {
                outCol = 0; // reset col on every new row
                for (int inCol = 0; inCol < inputs4D.cols() - filterWidth + 1; inCol += stride) {

                    // apply max pool filter
                    max = inputs4D.get(batchIdx, ch, inRow, inCol);
                    maxC = inCol;
                    maxR = inRow;
                    for (int fr = 0; fr < filterHeight; fr++) {
                        for (int fc = 0; fc < filterWidth; fc++) {
                            final float inputVal = inputs4D.get(batchIdx, ch, inRow + fr, inCol + fc);
                            if (inputVal > max) {
                                max = inputVal;
                                maxR = inRow + fr;
                                maxC = inCol + fc;
                            }
                        }
                    }

                    // zapamti indexe neurona iz prethodnog lejera koji su propustili max (koristice se u bacward pass-u)
                    maxIdxBatch[batchIdx][ch][outRow][outCol][0] = maxR; // height idx (row)
                    maxIdxBatch[batchIdx][ch][outRow][outCol][1] = maxC; // width idx (col)

                    outputs4D.set(max, batchIdx, ch, outRow, outCol); // set max value as output
                    outCol++;   // increase output col by one for each input (stride) step
                } // scan col
                outRow++; // increase output row by one for each input (stride) step
            } // scan row
        }
    }

    /**
     * backward pass for a max(x, y) operation has a simple interpretation as
     * only routing the gradient to the input that had the highest value in the
     * forward pass. Hence, during the forward pass of a pooling layer it is
     * common to keep track of the index of the max activation (sometimes also
     * called the switches) so that gradient routing is efficient during
     * backpropagation.
     *
     */
    @Override
    public void backward() {
        if (!isTrainable()) {
            return; // ovo ovde nije ni potrebno
        }

        if (DeepNetts.getInstance().useCuda()) {
            backwardImpl.backward();
        } else if (nextLayer instanceof FlattenLayer) {
            backwardFromFlatten();          
        } else if (nextLayer instanceof ConvolutionalLayer) {
            backwardFromConvolutional();
        }
        
    }

    private void backwardFromFlatten() {
        deltas.fill(0); // reset deltas to zero befor propagating deltas from next layer

        // ovo kreiranje ne treba ovde da se instancira u svakom prolazu nego u initu, a ovde samo poziv backwarda
        if (!multithreaded) {
            if (!batchMode) {
                backwardImpl = new SingleThreadedBackwardFromFlatten();
            } else {
                backwardImpl = new SingleThreadedBackwardBatchFromFlatten();
            }
        } else {
            backwardImpl = new MultiThreadedBackwardFromFlatten();
        }

        backwardImpl.backward();
    }

    private class SingleThreadedBackwardFromFlatten implements Backward {

        @Override
        public void backward() {
            Tensor3D deltas3d = (Tensor3D) deltas;
            for (int ch = 0; ch < deltas3d.depth(); ch++) { // zapravo isti broj kanala kao i ovaj layer
                backwardFromFlattenForChannel(ch);
            }
        }

    }

    private class SingleThreadedBackwardBatchFromFlatten implements Backward {

        @Override
        public void backward() {
            Tensor4D deltas4d = (Tensor4D) deltas;
            for (int ch = 0; ch < deltas4d.depth(); ch++) { // zapravo isti broj kanala kao i ovaj layer
                backwardFromFlattenForChannelBatch(ch);
            }
        }

    }

    private class MultiThreadedBackwardFromFlatten implements Backward {

        @Override
        public void backward() {
            try {
                threadPool.run(backwardTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage()); // throw excepti0on here!!!
            }
        }
    }

    // todo: ovaj prebaci u flatten - pa da propagira delte na prethodni - isti je za conv i maxpooling prev
    private void backwardFromFlattenForChannel(final int ch) {
        // ovaj u sustini treba da prkopira delte iz narednog flatten layera na odgovarajuce celije iz ovog
        Tensor1D nextDeltas1D = (Tensor1D) nextLayer.deltas;
        Tensor3D deltas3d = (Tensor3D) deltas;
        Tensor3D outputs3d = (Tensor3D) outputs;
        for (int row = 0; row < deltas3d.rows(); row++) {
            for (int col = 0; col < deltas3d.cols(); col++) { // ovaj povlaci delte direktno iz flatten layera
                final int ndIdx = row * outputs3d.cols() * outputs3d.depth() + col * outputs3d.depth() + ch; //!!! ovaj nije dobar                
                final float nextDelta = nextDeltas1D.get(ndIdx); // u deltas u flatten layeru su zapravo weighted deltas propagorani iza fc layera vidi flatten bacward
                deltas3d.set(nextDelta, ch, row, col); // set umesto add jer je vec sve sabrano u bacward od flatten - a batch??? trebalo bi da je to reseno u fc-u
            }
        }
    }

    private void backwardFromFlattenForChannelBatch(final int ch) {
        Tensor2D nextDeltas2D = (Tensor2D) nextLayer.deltas;
        Tensor4D deltas4d = (Tensor4D) deltas;
        Tensor4D outputs4d = (Tensor4D) outputs;
        for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            for (int row = 0; row < deltas4d.rows(); row++) {
                for (int col = 0; col < deltas4d.cols(); col++) { // ovaj povlaci delte direktno iz flatten layera
                    final int ndIdx = row * outputs4d.cols() * outputs4d.depth() + col * outputs4d.depth() + ch;
                    final float weightedDelta = nextDeltas2D.get(ndIdx, batchIdx); // u deltas u flatten layeru su zapravo weighted deltas propagorani iza fc layera vidi flatten bacward
                    deltas4d.set(weightedDelta, batchIdx, ch, row, col); // set umesto add jer je vec sve sabrano u bacward od flatten - a batch??? trebalo bi da je to reseno u fc-u
                }
            }
        }
    }

    private void backwardFromConvolutional() {
        deltas.fill(0);

        // ovo instanciranje prebaci u init
        if (!multithreaded) {
            if (!batchMode) {
                backwardImpl = new SingleThreadedBackwardFromConvolutional();
            } else {
                backwardImpl = new SingleThreadedBackwardBatchFromConvolutional();
            }
        } else {
            backwardImpl = new MultiThreadedBackwardFromConvolutional();
            // nemam multi threaded batch backward from convolutional
        }

        backwardImpl.backward();
    }

    private class SingleThreadedBackwardFromConvolutional implements Backward {

        @Override
        public void backward() {
            for (int ch = 0; ch < depth; ch++) { // iteriraj i 3-cu dimeziju sledeceg sloja odnosno kanale ovog sloja
                backwardFromConvolutionalForChannel(ch);
            }
        }
    }

    private class SingleThreadedBackwardBatchFromConvolutional implements Backward {

        @Override
        public void backward() {
            for (int ch = 0; ch < depth; ch++) { // iteriraj i 3-cu dimeziju sledeceg sloja odnosno kanale ovog sloja
                backwardFromConvolutionalForChannelBatch(ch);
            }
        }
    }

    private class MultiThreadedBackwardFromConvolutional implements Backward {

        @Override
        public void backward() {
            try {
                threadPool.run(backwardConvTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage()); // throw excepti0on here!!!
            }
        }

    }

    // fz je ch iz ovog lejera a treca diemnziaj filtera iz narednog lejera
    private void backwardFromConvolutionalForChannel(int fz) {
        final ConvolutionalLayer nextConvLayer = (ConvolutionalLayer) nextLayer;
        final int filterCenterX = (nextConvLayer.filterWidth - 1) / 2;
        final int filterCenterY = (nextConvLayer.filterHeight - 1) / 2;

        Tensor3D nextLayerDeltas = (Tensor3D) nextLayer.deltas;
        Tensor3D deltas3d = (Tensor3D) deltas;
        Tensor3D outputs3d = (Tensor3D) outputs;

        // 1. Propagate deltas from next conv layer for max outputs from this layer
        for (int ndz = 0; ndz < nextLayerDeltas.depth(); ndz++) { // iteriraj i 3-cu dimeziju sledeceg sloja odnosno kanale ovog sloja
            for (int ndr = 0; ndr < nextLayerDeltas.rows(); ndr++) { // sledeci lejer delte po visini
                for (int ndc = 0; ndc < nextLayerDeltas.cols(); ndc++) { // sledeci lejer delte po sirini
                    final float nextLayerDelta = nextLayerDeltas.get(ndz, ndr, ndc); // uzmi deltu iz sledeceg sloja za tekuci neuron (dx, dy, dz) sledeceg sloja

                    //   for (int fz = 0; fz < nextConvLayer.filterDepth; fz++) { umesto fz ide ch kao parametar
                    for (int fr = 0; fr < nextConvLayer.filterHeight; fr++) {
                        for (int fc = 0; fc < nextConvLayer.filterWidth; fc++) {
                            final int outRow = ndr * nextConvLayer.stride + (fr - filterCenterY);
                            final int outCol = ndc * nextConvLayer.stride + (fc - filterCenterX);

                            if (outRow < 0 || outRow >= outputs3d.rows() || outCol < 0 || outCol >= outputs3d.cols()) {
                                continue;
                            }

                            // deltas ima dosta medju nula
                            deltas3d.add(nextLayerDelta * nextConvLayer.filters.get(ndz, fz, fr, fc), fz, outRow, outCol); // da li se ovde preo z preklapaju?
                        }
                    }
                    // }
                }
            }
        }
        // FIX:
//           float divisor = nextConvLayer.filterWidth * nextConvLayer.filterHeight;  
//           deltas.div(divisor); // da li da delim sa dimenzijama filtera??? mnist radi bolje a cloud i cifar10 ne   ima slican efekat kao smanjivanje learning rate-a                
    }

    private void backwardFromConvolutionalForChannelBatch(final int fz) {
        final ConvolutionalLayer nextConvLayer = (ConvolutionalLayer) nextLayer;
        final int filterCenterX = (nextConvLayer.filterWidth - 1) / 2;
        final int filterCenterY = (nextConvLayer.filterHeight - 1) / 2;

        Tensor4D nextLayerDeltas = (Tensor4D) nextLayer.deltas;
        Tensor4D deltas4d = (Tensor4D) deltas;
        Tensor4D outputs4d = (Tensor4D) outputs;

        // 1. Propagate deltas from next conv layer for max outputs from this layer
        for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            for (int ndz = 0; ndz < nextLayerDeltas.depth(); ndz++) { // iteriraj i 3-cu dimeziju sledeceg sloja odnosno kanale ovog sloja
                for (int ndr = 0; ndr < nextLayerDeltas.rows(); ndr++) { // sledeci lejer delte po visini
                    for (int ndc = 0; ndc < nextLayerDeltas.cols(); ndc++) { // sledeci lejer delte po sirini
                        final float nextLayerDelta = nextLayerDeltas.get(batchIdx, ndz, ndr, ndc); // uzmi deltu iz sledeceg sloja za tekuci neuron (dx, dy, dz) sledeceg sloja

                        //   for (int fz = 0; fz < nextConvLayer.filterDepth; fz++) { umesto fz ide ch kao parametar
                        for (int fr = 0; fr < nextConvLayer.filterHeight; fr++) {
                            for (int fc = 0; fc < nextConvLayer.filterWidth; fc++) {
                                final int outRow = ndr * nextConvLayer.stride + (fr - filterCenterY);
                                final int outCol = ndc * nextConvLayer.stride + (fc - filterCenterX);

                                if (outRow < 0 || outRow >= outputs4d.rows() || outCol < 0 || outCol >= outputs4d.cols()) {
                                    continue;
                                }

                                // deltas ima dosta medju nula
                                deltas4d.add(nextLayerDelta * nextConvLayer.filters.get(ndz, fz, fr, fc), batchIdx, fz, outRow, outCol); // da li se ovde preo z preklapaju?
                            }
                        }

                    }
                }
            }
        }
    }

    /**
     * Does nothing for pooling layer since it does not have weights It just
     * propagates deltas from next layer to previous through connections that
     * had max activation in forward pass
     */
    @Override
    public void applyWeightChanges() {
    }

    public int getFilterWidth() {
        return filterWidth;
    }

    public int getFilterHeight() {
        return filterHeight;
    }

    public int getStride() {
        return stride;
    }

    /**
     * Calculates how many channels should be assigned in each thread in
     * multithreaded mode.
     *
     * @param threadCount
     * @return
     */
    private int[] calculateChannelsPerThread(int threadCount) {
        int[] threads = new int[threadCount];
        int chpt = depth / threadCount;

        for (int i = 0; i < threadCount; i++) {
            threads[i] = chpt;
        }

        if (depth % threadCount != 0) {
            int rest = depth % threadCount;

            for (int i = 0; i < rest; i++) {
                threads[i] = threads[i] + 1;
            }
        }

        return threads;
    }

    private class ForwardCallable implements Callable<Void> {

        final int fromCh, toCh;

        // Consumer<Integer> methodReference    https://dzone.com/articles/java-lambda-method-reference
        // mozda i da bude predikat da vraca boolean ukoliko je zavrsio, a false ako je prekinut
        public ForwardCallable(int fromCh, int toCh) {
            this.fromCh = fromCh;
            this.toCh = toCh;
        }

        @Override
        public Void call() throws Exception {

            forwardForChannelRange(fromCh, toCh);

            return null;
        }
    }

    private class BackwardFromConvolutionalCallable implements Callable<Void> {

        private final int fromCh, toCh;

        public BackwardFromConvolutionalCallable(int fromCh, int toCh) {
            this.fromCh = fromCh;
            this.toCh = toCh;
        }

        @Override
        public Void call() throws Exception {

            for (int ch = fromCh; ch < toCh; ch++) {
                backwardFromConvolutionalForChannel(ch);
            }

            return null;
        }
    }

    private class BackwardFromFlattenCallable implements Callable<Void> {

        private final int fromCh, toCh;

        public BackwardFromFlattenCallable(int fromCh, int toCh) {
            this.fromCh = fromCh;
            this.toCh = toCh;
        }

        @Override
        public Void call() throws Exception {

            for (int ch = fromCh; ch < toCh; ch++) {
                backwardFromFlattenForChannel(ch);
            }

            return null;
        }
    }

    public float getL1WeightSum() {
        return 0f;
    }

    public float getL2WeightSum() {
        return 0f;
    }

    @Override
    public String toString() {
        return "Max Pooling Layer { filter width:" + filterWidth + ", filter height: " + filterHeight + ", stride:" + stride + "}";
    }

}
