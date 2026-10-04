/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 * This file is part of DeepNetts.
 *
 */
package deepnetts.net.layers;

import deepnetts.net.layers.activation.ActivationType;
import deepnetts.core.DeepNetts;
import deepnetts.util.DeepNettsException;
import deepnetts.net.weights.RandomWeights;
import deepnetts.tensor.TensorBase;
import java.util.logging.Logger;
import deepnetts.net.layers.activation.MathFunctions;
import deepnetts.net.train.opt.Optimizer;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.Tensors;
import deepnetts.util.CallableRangeConsumer;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

/**
 * Convolutional layer performs image convolution operation on outputs of a
 * previous layer using filters. This filtering operation is similar like
 * applying image filters in photoshop, but this filters can also be trained to
 * learn image features of interest.
 *
 * Layer include parameters: filter's width, heigh Number of filters / depth
 * Step when applying filters : stride Padding, which is an image border to keep
 * the size of image and avoid information loss padding Stride defaults to 1
 *
 */
public final class ConvolutionalLayer extends AbstractLayer<TensorBase, TensorBase, Tensor4D> {

    private static final long serialVersionUID = -3972836675081087082L;

    /**
     * Convolutional filters. Filters are stored as tensors.
     */
    Tensor4D filters;

    /**
     * Convolutional filter width (columns)
     */
    final int filterWidth;

    /**
     * Filter height (rows)
     */
    final int filterHeight;

    /**
     * Filter depth, corresponds to number of channels in previous layer.
     */
    int filterDepth; // initialized when building network

    // final int filterRadius;
    int filterGroups = 1; // 2
    int inGroupSize = -1;
    int outGroupSize = -1;

    /**
     * Convolution step, 1 by default. Number of steps convolutional filter is
     * moved during convolution. Commonly used values 1, 2, rarely 3
     */
    int stride = 1;

    /**
     * Border padding filled with zeros (0, 1 or 2) Usually half of the filter
     * size
     */
    int padding = 0;

    int fCenterX; //  padding = (kernel-1)/2
    int fCenterY;

    int[][][][] maxIdx;
    int[][][][][] maxIdxBatch;

    private transient List<Callable<Void>> forwardTasks;
    private transient List<Callable<Void>> backwardFromPoolingTasks;
    private transient List<Callable<Void>> backwardFromConvolutionalTasks;
    private transient List<Callable<Void>> backwardFromFlattenTasks;
    private boolean multithreaded = false;

//    private transient Forward forwardComputation;
//    private transient Backward backwardComputation;

    private static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());

    /**
     * Create a new instance of convolutional layer with specified number of
     * channels filter size, default padding (filter-1)/2, and default stride
     * stride value 1, and specified number of channels. Uses Linear activation
     * function by default.
     *
     * @param channels number of channels which corresponds to the number of
     * image pixel feature that you want to learn and detect
     * @param filterWidth
     * @param filterHeight
     */
    public ConvolutionalLayer(int channels, int filterWidth, int filterHeight) {
        // sve mora da bude pozitivno. filteri motaju da budu  neparni - validacija
        super(ActivationType.LINEAR);

        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        //   this.filterRadius = (filterWidth - 1) / 2; // assumes same x and y radius 
        this.depth = channels; // ovo je isto kao i depth, broj feature mapa
        this.filterGroups = 1;
        this.stride = 1;
    }

    public ConvolutionalLayer(int channels, Filter filter) {
        super(ActivationType.LINEAR);
        // sve mora da bude pozitivno. filteri motaju da budu  neparni - validacija
        this.filterWidth = filter.getWidth();
        this.filterHeight = filter.getHeight();
        //      this.filterRadius = (filterWidth - 1) / 2; // assumes same x and y radius 
        this.depth = channels; // ovo je isto kao i depth, broj feature mapa
        this.filterGroups = filter.getGroups();
        this.stride = filter.getStride();
    }

    public ConvolutionalLayer(int channels, Filter filter, ActivationType activationType) {
        super(activationType);
        this.filterWidth = filter.getWidth();
        this.filterHeight = filter.getHeight();
        //   this.filterRadius = (filterWidth - 1) / 2; // assumes same x and y radius 
        this.depth = channels;
        this.filterGroups = filter.getGroups();
        this.stride = filter.getStride();
    }

    public ConvolutionalLayer(int channels, int filterWidth, int filterHeight, int stride, ActivationType activationType) {
        super(activationType);
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        //    this.filterRadius = (filterWidth - 1) / 2; // assumes same x and y radius 
        this.depth = channels;
        this.stride = stride;
    }

    /**
     * Initialize dimensions, create output tensors, filters/weights, biases and
     * all internal structures etc.
     *
     * Assumes that prevLayer is set in network builder
     */
    @Override
    public void init() {
        // prev layer can only be input, max pooling or convolutional
        if (!(prevLayer instanceof InputLayer || prevLayer instanceof ConvolutionalLayer || prevLayer instanceof MaxPoolingLayer)) {
            throw new DeepNettsException("Illegal architecture: convolutional layer can be used only after input, convolutional or maxpooling layer");
        }

        inputs = prevLayer.getOutputs();

        width = (prevLayer.getWidth()) / stride;
        height = (prevLayer.getHeight()) / stride;
        // depth is set in constructor

        // ovi mogu da budu final ubaci ih u konstruktore
        fCenterX = (filterWidth - 1) / 2; //  padding = filter /2
        fCenterY = (filterHeight - 1) / 2;
        // todo set padding ! padding =
        this.padding = fCenterX;

        if (prevLayer.depth % filterGroups != 0) {
            throw new DeepNettsException("Number of input channels must be divisible by number of filter groups!");
        }
        inGroupSize = prevLayer.depth / filterGroups; // velicina grupe (broj kanala) u inputu

        if (depth % filterGroups != 0) {
            throw new DeepNettsException("Number of output channels must be divisible by number of filter groups!");
        }
        outGroupSize = depth / filterGroups; // velicina grupe / broj kanala u outputu == broj filtera u grupi

        // if inputs are Tensor4D it's a batch mode
        batchMode = (inputs instanceof Tensor4D);

        // init output cells, deltas and derivative buffer
        if (!batchMode) {
            outputs = new Tensor3D(depth, height, width); // ovo je rows cols channels
            deltas = new Tensor3D(depth, height, width);
        } else { // kako da znam da li je batch mode
            batchSize = ((Tensor4D) inputs).fourthDim();
            outputs = new Tensor4D(batchSize, depth, height, width); // ovo je rows cols channels
            deltas = new Tensor4D(batchSize, depth, height, width);
        }

        // init filters(weights) - broj filtera je isti kao i broj kanala/dubina prethodnog lejera
        filterDepth = prevLayer.getDepth(); // ne mora da bude, nego neka bude isti broj koliko i channels / layer depth

        int inputCount = filterWidth * filterHeight * filterDepth; // + 1 for bias , mozda bias ne racunati jer se on nezavisno inicijalizuje

        // tf format weights[kernel_height, kernel_width, kernel_depth, ch]
        // da bi islo brze mislim d atreba depth/ch, filterDepth, filterHeight, filterWidth - vidi kako trazi cudnn
        filters = new Tensor4D(depth, filterDepth, filterHeight, filterWidth); // depth bi bilo bolje da bud eprvi da brze iterira ali nije kriticno trenutno

        RandomWeights.uniform(filters.getValues(), inputCount);
        // TODO: srediti radnomizaciju conv layera! suma bi trebalo da bude nula? kao pravi image filter, nekako normlizovan
        // RandomWeights.gaussian(filters.getValues(), 0, (float)Math.sqrt(2.0f/inputCount));  // ovde treba da ide he randomizacija sa relu
        // suma treba da bude nula, kao kod image filtera svaki pojedinacni filter da ima gausovu disttribuciju
        gradients = new Tensor4D(depth, filterDepth, filterHeight, filterWidth);
        deltaWeights = new Tensor4D(depth, filterDepth, filterHeight, filterWidth);
        prevDeltaWeights = new Tensor4D(depth, filterDepth, filterHeight, filterWidth);

        // and biases               // svaki kanal ima svoj filter i svoj bias - sta ako prethodni sloj ima vise biasa? mislim da bi tada svaki filter trebalo da ima svoj bias ovo bi znaci trebalo da bude 2D biases[depth][prevLayerDepth]
        biases = new Tensor1D(depth);
        deltaBiases = new Tensor1D(depth);
        prevDeltaBiases = new Tensor1D(depth);

        if (activationType == ActivationType.RELU) {
            biases.fill(0.1f); // good for relu - za tanh 0
        } else {
            //biases.fill(0f);// proveriti ovo, da li randomizovati ili sta..., ne bih ja stavljao nulu... mada mozda i moze jer ce u prvoj iteraciji inijalizovati na gradijent...
            RandomWeights.gaussian(biases.getValues(), 0.1f, 0.05f);
        }
        //temporary hack        
        optimizer = Optimizer.create(OptimizerType.SGD, this); // default optimizer        

        this.weights = filters; // zbog buga u testovima npe
        initTransientFields();
    }

    @Override
    public void initTransientFields() {
        int threadCount = Math.min(DeepNetts.getInstance().getMaxThreads(), depth); // ovo zameni 
        if (threadCount > 1) {
            multithreaded = true;
            int[] channelsPerThread = calculateChannelsPerThread(threadCount); // ovo zameni da ima neki min broj operacija za svaki thread

            forwardTasks = new ArrayList<>();
            backwardFromPoolingTasks = new ArrayList<>();
            backwardFromConvolutionalTasks = new ArrayList<>();
            backwardFromFlattenTasks = new ArrayList<>();

            int fromCh = 0, toCh = 0;
            for (int i = 0; i < threadCount; i++) {
                fromCh = toCh;
                toCh = fromCh + channelsPerThread[i];

                if (!batchMode) {
                    CallableRangeConsumer ftask = new CallableRangeConsumer(fromCh, toCh, this::forwardForChannelRange);
                    forwardTasks.add(ftask);
                } else {
                    CallableRangeConsumer ftask = new CallableRangeConsumer(fromCh, toCh, this::forwardForChannelRangeBatch); // moze i ovde CallableRangeConsumer
                    forwardTasks.add(ftask);
                }

                // ovde svuda ispod moze CallableRangeConsumer umesto CallableIntConsumer
                if (nextLayer instanceof MaxPoolingLayer) {
                    CallableRangeConsumer btask = new CallableRangeConsumer(fromCh, toCh, this::backwardFromMaxPoolingForChannelRange);
                    backwardFromPoolingTasks.add(btask);
                } else if (nextLayer instanceof ConvolutionalLayer) {
                    CallableRangeConsumer bctask = new CallableRangeConsumer(fromCh, toCh, this::backwardFromConvolutionalForChannel);
                    backwardFromConvolutionalTasks.add(bctask);
                } else if (nextLayer instanceof FlattenLayer) {
                    CallableRangeConsumer bftask = new CallableRangeConsumer(fromCh, toCh, this::backwardFromFlattenForChannels);
                    backwardFromFlattenTasks.add(bftask);
                }
            }
        }

        if (DeepNetts.getInstance().useCuda()) {
         //   forward = new ConvolutionalForwardCuda(cudaHandles, this);
         //   forwardImpl = AcceleratorService.defaultProvider().createConvolutionalForwardAcc(cudaHandles, this);
           // backward = new ConvolutionalBackwardCuda(cudaHandles, this);
         //   backwardImpl = AcceleratorService.defaultProvider().createConvolutionalBackwardAcc(cudaHandles, this);
        } else if (!multithreaded) { // single threaded
            if (!batchMode) { // inputs instanceof Tensor1D
                forwardImpl = new SingleThreadedForward(); // default forward
            } else if (batchMode) { // inputs instanceof Tensor2D , resi i slucaj za Tensor4D to je na GPU
                forwardImpl = new SingleThreadedForwardBatch();
            }
        } else { // multithreaded
            forwardImpl = new MultiThreadedForward();
            //backwardComputation = new MultiThreadedBackward();
        }
    }

    /**
     * Forward pass for convolutional layer. Performs convolution operation on
     * inputs (output from previous layer) using filters in this layer, on all
     * channels. Each channel from prev layer has its own filter (3D filter),
     * and every channel in this layer has its 3D filter used to scan all
     * channels in prev layer.
     *
     * Previous layers can be: Input, MaxPooling or Convolutional.
     *
     * For more about convolution see
     * http://www.songho.ca/dsp/convolution/convolution.html
     */
    @Override
    public void forward() {
        // hack, move elsvere, mozda mora ovde ako je prethodni layer input
        inputs = prevLayer.getOutputs();

        forwardImpl.forward();        
        
        // cuda formard impl 
//        if (DeepNetts.getInstance().useCuda()) {
//            forwardComputation.forward();
//            return;
//        } else {
//            forwardComputation.forward();
//        }

//        if (!multithreaded) {
//            if (batchMode) {
//                 // a single threaded batch - does not make sense but required for debugging
//                 forwardForBatchRange(0, batchSize); // sta je vece batch ili broj kanala
//                 // broj kanala moze biti 1000 a i broj slika
//                 // sta je veci posao 
//                 // sta ako imam 100 threadova
//            } else {
//                forwardForChannelRange(0, this.depth);
//            }
//        } else {
//            try {
//               threadPool.run(forwardTasks);
//            } catch (InterruptedException ex) {
//                LOG.warning(ex.getMessage());
//            }
//        }

    }

    private class SingleThreadedForward implements Forward {

        @Override
        public void forward() {
            forwardForChannelRange(0, depth);
        }

    }

    private class SingleThreadedForwardBatch implements Forward {

        @Override
        public void forward() {
            int fromBatch = 0, toBatch = batchSize;

            final Tensor4D inputs4d = (Tensor4D) inputs;
            final Tensor4D outputs4d = (Tensor4D) outputs;

            final int inputsRows = inputs4d.rows();
            final int inputsCols = inputs4d.cols();

            for (int batchIdx = fromBatch; batchIdx < toBatch; batchIdx++) {  // i to za outputs deltas i inputs
                for (int ch = 0; ch < depth; ch++) {
                    int outRow = 0, outCol = 0; // reset indexes for current output's row and col
                    for (int inRow = 0; inRow < inputsRows; inRow += stride) { // iterate all input rows
                        outCol = 0; // every time when input goes in next row, output does too, so reset column idx

                        for (int inCol = 0; inCol < inputsCols; inCol += stride) { // iterate all input cols
                            outputs4d.set(biases.get(ch), batchIdx, ch, outRow, outCol); // sum will be added to bias - I can set entire matrix to bias initial values  above

                            int groupIdx = ch / outGroupSize; // mislim da je to to, celobrojno deljenje zavrsava posao
                            final int fzStart = 0; //  groupIdx * inGroupSize; // TODO: za aktiviranje groups
                            final int fzEnd = filterDepth; // fzStart + inGroupSize; TODO: za aktiviranje groups

                            float filterOut = 0;
                            float filterRadius = (filterWidth - 1) / 2;

                            if (inCol >= filterRadius && inCol < inputsCols - filterRadius - 1
                                    && inRow >= filterRadius && inRow < inputsRows - filterRadius - 1) {
                                filterOut = applyFilterInTheMiddleBatch(batchIdx, ch, inRow, inCol); // ovaj je radio 29.12. za vgg
                            } else {
                                // filterOut = applyFilter(inRow, inCol, ch);
                                filterOut = applyFilterOnEdgeBatch(batchIdx, ch, inRow, inCol);
                            }

                            outputs4d.add(filterOut, batchIdx, ch, outRow, outCol);
                            outCol++; // move to next col in out layer after each filter position
                        } // iterate cols
                        outRow++; // every time input goes to next row (inR), output does too
                    } // iterate rows
                }//iterate channels
                // apply activation function on current channel
                activation.apply(outputs4d, batchIdx); // ovaj treba promeniti ch i batch           
            } // batch
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

    /**
     * Performs forward pass calculation for specified channel. Forward
     * convolution operation takes input from all channels in previous layer.
     *
     * @param ch channel to calculate
     */
    private void forwardForChannelRange(final int fromCh, final int toCh) {

        final Tensor3D inputs3d = (Tensor3D) inputs;
        final Tensor3D outputs3d = (Tensor3D) outputs;

        final int inputsRows = inputs3d.rows();
        final int inputsCols = inputs3d.cols();

        for (int ch = fromCh; ch < toCh; ch++) {
            int outRow = 0, outCol = 0; // reset indexes for current output's row and col
            for (int inRow = 0; inRow < inputsRows; inRow += stride) { // iterate all input rows
                outCol = 0; // every time when input goes in next row, output does too, so reset column idx

                for (int inCol = 0; inCol < inputsCols; inCol += stride) { // iterate all input cols
                    outputs3d.set(biases.get(ch), ch, outRow, outCol); // sum will be added to bias - I can set entire matrix to bias initial values  above

                    int groupIdx = ch / outGroupSize; // mislim da je to to, celobrojno deljenje zavrsava posao

                    final int fzStart = 0; //  groupIdx * inGroupSize; // TODO: za aktiviranje groups
                    final int fzEnd = filterDepth; // fzStart + inGroupSize; TODO: za aktiviranje groups

                    float filterOut = 0;
                    float filterRadius = (filterWidth - 1) / 2;

                    if (inCol >= filterRadius && inCol < inputsCols - filterRadius - 1
                            && inRow >= filterRadius && inRow < inputsRows - filterRadius - 1) {
                        filterOut = applyFilterInTheMiddle(ch, inRow, inCol); // ovaj je radio 29.12. za vgg
                    } else {
                        filterOut = applyFilterOnEdge(ch, inRow, inCol);
                    }

                    outputs3d.add(filterOut, ch, outRow, outCol);
                    outCol++; // move to next col in out layer after each filter position
                } // iterate cols
                outRow++; // every time input goes to next row (inR), output does too
            } // iterate rows

            // apply activation function on current channel
            activation.apply(outputs3d, ch);
        }
    }

    private void forwardForChannelRangeBatch(final int fromCh, final int toCh) {
        final Tensor4D inputs4d = (Tensor4D) inputs;
        final Tensor4D outputs4d = (Tensor4D) outputs;

        final int inputsRows = inputs4d.rows();
        final int inputsCols = inputs4d.cols();

        for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {  // i to za outputs deltas i inputs
            for (int ch = fromCh; ch < toCh; ch++) {
                int outRow = 0, outCol = 0; // reset indexes for current output's row and col
                for (int inRow = 0; inRow < inputsRows; inRow += stride) { // iterate all input rows
                    outCol = 0; // every time when input goes in next row, output does too, so reset column idx

                    for (int inCol = 0; inCol < inputsCols; inCol += stride) { // iterate all input cols
                        outputs4d.set(biases.get(ch), batchIdx, ch, outRow, outCol); // sum will be added to bias - I can set entire matrix to bias initial values  above

                        int groupIdx = ch / outGroupSize; // mislim da je to to, celobrojno deljenje zavrsava posao
                        final int fzStart = 0; //  groupIdx * inGroupSize; // TODO: za aktiviranje groups
                        final int fzEnd = filterDepth; // fzStart + inGroupSize; TODO: za aktiviranje groups

                        float filterOut = 0;
                        float filterRadius = (filterWidth - 1) / 2;

                        if (inCol >= filterRadius && inCol < inputsCols - filterRadius - 1
                                && inRow >= filterRadius && inRow < inputsRows - filterRadius - 1) {
                            filterOut = applyFilterInTheMiddle(ch, inRow, inCol); // ovaj je radio 29.12. za vgg
                        } else {
                            // filterOut = applyFilter(inRow, inCol, ch);
                            filterOut = applyFilterOnEdge(ch, inRow, inCol);
                        }

                        outputs4d.add(filterOut, batchIdx, ch, outRow, outCol);
                        outCol++; // move to next col in out layer after each filter position
                    } // iterate cols
                    outRow++; // every time input goes to next row (inR), output does too
                } // iterate rows
            }//iterate channels
            // apply activation function on current channel
            activation.apply(outputs4d, batchIdx); // ovaj treba promeniti ch i batch           
        } // batch
    }

    // bolje da prefromulisem kako da u chunki obradim sve ove u multithreaded 
    // preradi ga sa ovim novim forward!!! forwardForChannelRange ako ima 512 kanala kako vggnet?
    // forwardForChannelRange
    private void forwardForBatchRange(final int fromBatch, final int toBatch) {
        final Tensor4D inputs4d = (Tensor4D) inputs;
        final Tensor4D outputs4d = (Tensor4D) outputs;

        final int inputsRows = inputs4d.rows();
        final int inputsCols = inputs4d.cols();

        for (int batchIdx = fromBatch; batchIdx < toBatch; batchIdx++) {  // i to za outputs deltas i inputs
            for (int ch = 0; ch < depth; ch++) {
                int outRow = 0, outCol = 0; // reset indexes for current output's row and col
                for (int inRow = 0; inRow < inputsRows; inRow += stride) { // iterate all input rows
                    outCol = 0; // every time when input goes in next row, output does too, so reset column idx

                    for (int inCol = 0; inCol < inputsCols; inCol += stride) { // iterate all input cols
                        outputs4d.set(biases.get(ch), batchIdx, ch, outRow, outCol); // sum will be added to bias - I can set entire matrix to bias initial values  above

                        int groupIdx = ch / outGroupSize; // mislim da je to to, celobrojno deljenje zavrsava posao
                        final int fzStart = 0; //  groupIdx * inGroupSize; // TODO: za aktiviranje groups
                        final int fzEnd = filterDepth; // fzStart + inGroupSize; TODO: za aktiviranje groups

                        float filterOut = 0;
                        float filterRadius = (filterWidth - 1) / 2;

                        if (inCol >= filterRadius && inCol < inputsCols - filterRadius - 1
                                && inRow >= filterRadius && inRow < inputsRows - filterRadius - 1) {
                            filterOut = applyFilterInTheMiddleBatch(batchIdx, ch, inRow, inCol); // ovaj je radio 29.12. za vgg
                        } else {
                            // filterOut = applyFilter(inRow, inCol, ch);
                            filterOut = applyFilterOnEdgeBatch(batchIdx, ch, inRow, inCol);
                        }

                        outputs4d.add(filterOut, batchIdx, ch, outRow, outCol);
                        outCol++; // move to next col in out layer after each filter position
                    } // iterate cols
                    outRow++; // every time input goes to next row (inR), output does too
                } // iterate rows
            }//iterate channels
            // apply activation function on current channel
            activation.apply(outputs4d, batchIdx); // ovaj treba promeniti ch i batch           
        } // batch
    }

    // ovom metodom zameni kod iznad, a zatim dodaj jos jednu metodu kojom ces da zamenis ovu u slucaju po sredini i onda profajliraj i izbroj broj poziva
    // ostalo je jos da se optimizuju geteri iz ove metode! da i ovde radi sa buferima i vektorizacijom kao i u sredini - jedan po jedan slucaj za svih 8 pozicija if
    private float applyFilter(int inRow, int inCol, int ch) {

        final Tensor3D inputs3d = (Tensor3D) inputs;

        final int inputsRows = inputs3d.rows();
        final int inputsCols = inputs3d.cols();
        final int filterColStart = inCol - fCenterX;
        final int filterRowStart = inRow - fCenterY;
        final int fzStart = 0; //  groupIdx * inGroupSize; // TODO: za aktiviranje groups
        final int fzEnd = filterDepth; // fzStart + inGroupSize; TODO: za aktiviranje groups        filterColStart = inCol - fCenterX;

        float filterOut = 0;

        for (int fz = fzStart; fz < fzEnd; fz++) { // iterate filter by depth - number of channels in previous layer (input shannels)
            for (int fr = 0; fr < filterHeight; fr++) { // iterate filter by height/rows
                final int fcr = filterRowStart + fr; // filter center row positionb- i ovo optimizuj 2 operaije umesto 3 oduzimanje je uvek isto
                for (int fc = 0; fc < filterWidth; fc++) { // iterate filter by width / columns
                    final int fcc = filterColStart + fc; // filter center col position

                    // skip input indexes which are out of bounds - ovaj uslov pojednostavim. Mozda da idem od -filterWidth/2 do filterWidth/2
                    if (fcr < 0 || fcr >= inputsRows || fcc < 0 || fcc >= inputsCols) { // proveri kako ovo pretvara u bytecode!
                        continue;
                    }
                    // fz - prilagodi group. input zavisi od filter deptha
                    filterOut += inputs3d.get(fz, fcr, fcc) * filters.get(ch, fz, fr, fc); // i generic za inputs
                }
            }
        }
        return filterOut;
    }

    // todo: ovaj treba ubrzati!
    // otpruilike je 2-3x veci od vectorized in the middle, a polovina vremena otpada na getere
    private float applyFilterOnEdge(int ch, int inRow, int inCol) {

        final Tensor3D inputs3d = (Tensor3D) inputs;

        final int inputsRows = inputs3d.rows();
        final int inputsCols = inputs3d.cols();

        // svo ovo odredjivanje indeksa izmestiti iz ove metode tako da se radi samo jednim
        // in fow start racunati u pow petlji i prosledjivati kao parametar, celu metodu inlajnovati, da ne racuna za svaku celiju nego samo jednom za ceo red
        // dodatno moze se glavna petlja razdvojiti na gornji deo, sredinu i donji da eliminisem sve if-ove
        int filterRowStart = 0;
        int inRowStart = inRow - fCenterY;
        if (inRowStart < 0) {
            filterRowStart = Math.abs(inRowStart); //onoliko koliko je otisao u minus treba pomeriti pocetak indeksa u filteru
            inRowStart = 0;
        }

        int inRowEnd = inRowStart + (filterHeight - filterRowStart); // ovaj nije dobar treba ga skratiti z aonoliko koliko je otisao u minus ako je otisao u minus
        if (inRowEnd > inputsRows) {
            inRowEnd = inputsRows;
        }

        int filterColStart = 0;
        int inColStart = inCol - fCenterX;
        if (inColStart < 0) {
            filterColStart = Math.abs(inColStart);
            inColStart = 0;
        }

        int inColEnd = inColStart + (filterWidth - filterColStart);
        if (inColEnd > inputsCols) {
            inColEnd = inputsCols;
        }

        final int fzStart = 0; //  groupIdx * inGroupSize; // TODO: za aktiviranje groups
        final int fzEnd = filterDepth; // fzStart + inGroupSize; TODO: za aktiviranje groups        filterColStart = inCol - fCenterX;

        float filterOut = 0;
        for (int fz = fzStart; fz < fzEnd; fz++) { // iterate filter by depth - number of channels in previous layer (input shannels)
            for (int ir = inRowStart, fr = filterRowStart; ir < inRowEnd; ir++, fr++) { // iterate filter by height/rows
                // zameni ovu jednu for petlju sa array copy i mnozenjem vektora - moze kad je row first layout
                // to ce eliminisati i getere 
                for (int ic = inColStart, fc = filterColStart; ic < inColEnd; ic++, fc++) { // iterate filter by width / columns
                    filterOut += inputs3d.get(fz, ir, ic) * filters.get(ch, fz, fr, fc); // ovde treba za filtere zero based
                    //inputs.getSubArray(ir, inColStart, inColEnd  fz) filters.get(fr, filterColStart, filterColEnd, fz, ch)
                }
            }
        }
        return filterOut;
    }

    private float applyFilterOnEdgeBatch(int batchIdx, int ch, int inRow, int inCol) {

        final Tensor4D inputs4d = (Tensor4D) inputs;

        final int inputsRows = inputs4d.rows();
        final int inputsCols = inputs4d.cols();

        // iterate this in batch
        // svo ovo odredjivanje indeksa izmestiti iz ove metode tako da se radi samo jednim
        // in fow start racunati u pow petlji i prosledjivati kao parametar, celu metodu inlajnovati, da ne racuna za svaku celiju nego samo jednom za ceo red
        // dodatno moze se glavna petlja razdvojiti na gornji deo, sredinu i donji da eliminisem sve if-ove
        int filterRowStart = 0;
        int inRowStart = inRow - fCenterY;
        if (inRowStart < 0) {
            filterRowStart = Math.abs(inRowStart); //onoliko koliko je otisao u minus treba pomeriti pocetak indeksa u filteru
            inRowStart = 0;
        }

        int inRowEnd = inRowStart + (filterHeight - filterRowStart); // ovaj nije dobar treba ga skratiti z aonoliko koliko je otisao u minus ako je otisao u minus
        if (inRowEnd > inputsRows) {
            inRowEnd = inputsRows;
        }

        int filterColStart = 0;
        int inColStart = inCol - fCenterX;
        if (inColStart < 0) {
            filterColStart = Math.abs(inColStart);
            inColStart = 0;
        }

        int inColEnd = inColStart + (filterWidth - filterColStart);
        if (inColEnd > inputsCols) {
            inColEnd = inputsCols;
        }

        final int fzStart = 0; //  groupIdx * inGroupSize; // TODO: za aktiviranje groups
        final int fzEnd = filterDepth; // fzStart + inGroupSize; TODO: za aktiviranje groups        filterColStart = inCol - fCenterX;

        float filterOut = 0;
        for (int fz = fzStart; fz < fzEnd; fz++) { // iterate filter by depth - number of channels in previous layer (input shannels)
            for (int ir = inRowStart, fr = filterRowStart; ir < inRowEnd; ir++, fr++) { // iterate filter by height/rows
                // zameni ovu jednu for petlju sa array copy i mnozenjem vektora - moze kad je row first layout
                // to ce eliminisati i getere 
                for (int ic = inColStart, fc = filterColStart; ic < inColEnd; ic++, fc++) { // iterate filter by width / columns
                    filterOut += inputs4d.get(batchIdx, fz, ir, ic) * filters.get(ch, fz, fr, fc); // ovde treba za filtere zero based
                }
            }
        }

        return filterOut;
    }

    // primer: width=32  filter = 3 filter radius=1    32-1-1=30 ; poslednja pozicija za filter 3 je 
    // -1 je filter radius i jos jedan -1 jer pocinje od nule
    // if inCol >= filterRadius && inCol < inputWidth-filterRadius-1 && proveri gornju granicu
    //     inRow >= filterRadiues && inRow < inputHeight-filterRadius-1
    // izbacen if iz petlje primenjivo samo po sredini kanala
    private float applyFilterInTheMiddle(final int ch, final int inRow, final int inCol) {
        final Tensor3D inputs3d = (Tensor3D) inputs;

        float filterOut = 0;
        final int inColStart = inCol - fCenterX;
        final int inRowStart = inRow - fCenterY;

        for (int fz = 0; fz < filterDepth; fz++) { // iterate filter by depth - number of channels in previous layer (input shannels)
            for (int fr = 0; fr < filterHeight; fr++) { // iterate filter by height/rows
                final int fir = inRowStart + fr; // filtered input row    
                for (int fc = 0; fc < filterWidth; fc++) { // iterate filter by width / columns -- zameni ovo sa array copy ceo row vektor iz inputa i filtera
                    final int fic = inColStart + fc; // filtered input col umn
                    filterOut += inputs3d.get(fz, fir, fic) * filters.get(ch, fz, fr, fc);
                }
            }
        }
        return filterOut;
    }

    private float applyFilterInTheMiddleBatch(final int batchIdx, final int ch, final int inRow, final int inCol) {
        final Tensor4D inputs4d = (Tensor4D) inputs;

        float filterOut = 0;
        final int inColStart = inCol - fCenterX;
        final int inRowStart = inRow - fCenterY;

        for (int fz = 0; fz < filterDepth; fz++) { // iterate filter by depth - number of channels in previous layer (input shannels)
            for (int fr = 0; fr < filterHeight; fr++) { // iterate filter by height/rows
                final int fir = inRowStart + fr; // filtered input row    
                for (int fc = 0; fc < filterWidth; fc++) { // iterate filter by width / columns -- zameni ovo sa array copy ceo row vektor iz inputa i filtera
                    final int fic = inColStart + fc; // filtered input col umn
                    filterOut += inputs4d.get(batchIdx, fz, fir, fic) * filters.get(ch, fz, fr, fc);
                }
            }
        }
        return filterOut;
    }

    boolean useConvCache = true;
    boolean filterCacheCreated = false;
    float[][] filterCache;

    public void createFilterCache() {

        if (filterCache == null) {
            filterCache = new float[depth][filters.depth() * filters.rows() * filters.cols()]; // treba samo za ovaj kanal            
        }
        // filter se jednom kesira ali input? neka prvi prolaz kesira za sve ostale? input se uvek menja - razmotri to ali makar filter eliminisi\
        // napravi cache tako da se lako prekopir iz originalno tenzora, mozda po cela kolonau duzini filtera

        for (int ch = 0; ch < depth; ch++) {
            int fCacheIdx = 0;
            for (int fz = 0; fz < filterDepth; fz++) { // iterate filter by depth - number of channels in previous layer (input shannels)
                for (int fr = 0; fr < filterHeight; fr++) { // iterate filter by height/rows - zameni samo redosled ovih i extra ce biti
                    for (int fc = 0; fc < filterWidth; fc++) { // iterate filter by width / columns -- zameni ovo sa array copy ceo row vektor iz inputa i filtera
                        filterCache[ch][fCacheIdx] = filters.get(ch, fz, fr, fc);
                        fCacheIdx++;
                    }
                }
            }
        }
        filterCacheCreated = true;// kad prvi zavrsi i kreira filter cache to utice i na sve ostale
    }

    private float applyFilterInTheMiddleWithCaching(final int ch, final int inRow, final int inCol) {
        float filterOut = 0;
        final int inColStart = inCol - fCenterX;
        final int inRowStart = inRow - fCenterY;

        final Tensor3D inputs3d = (Tensor3D) inputs;

        // kada kreiram ovaj caches in kako ih resavam za multithreading?
        // kreiraj filter cache pre formwarda, ovo jekorsiti samo za inference mode - da se optimizuje samo za forward pass
        // ovde je sad problem multi threadinga kako da threadovi ne gaze jedan drugom cache?  mora svaki thread da ima svoj cache - resi to ond atetsiraj
//            if (filterCache == null) {
//                filterCache = new float[depth][filters.depth() * filters.rows() * filters.cols()]; // treba samo za ovaj kanal            
//            }
        // filter se jednom kesira ali input? neka prvi prolaz kesira za sve ostale? input se uvek menja - razmotri to ali makar filter eliminisi\
        // napravi cache tako da se lako prekopir iz originalno tenzora, mozda po cela kolonau duzini filtera
        int cacheIdx = 0;

        for (int fz = 0; fz < filterDepth; fz++) { // iterate filter by depth - number of channels in previous layer (input shannels)
            for (int fr = 0; fr < filterHeight; fr++) { // iterate filter by height/rows - zameni samo redosled ovih i extra ce biti
                final int fir = inRowStart + fr; // filtered input row    
                for (int fc = 0; fc < filterWidth; fc++) { // iterate filter by width / columns -- zameni ovo sa array copy ceo row vektor iz inputa i filtera
                    final int fic = inColStart + fc; // filtered input col umn
                    if (!filterCacheCreated) {
                        filterOut += inputs3d.get(fz, fir, fic) * filters.get(ch, fz, fr, fc);
                        filterCache[ch][cacheIdx] = filters.get(ch, fz, fr, fc);
                        //                       inputCache[ch][inRow][inCol][cacheIdx] = inputs.get(fz, fir, fic); // input ne moze da se kesira jer se stalno menja
                    } else {
                        filterOut += inputs3d.get(fz, fir, fic) * filterCache[ch][cacheIdx]; // ovdce svi threadovi jedn idrugima gaze cache - svaki mora da ima svoj cache
                    }

                    cacheIdx++;
                }
            }
        }
        filterCacheCreated = true;// kad prvi zavrsi i kreira filter cache to utice i na sve ostale

        return filterOut;
    }

    /**
     * Primeni filter na sredini kanala gde nema odsecanja Pozicija centra
     * filtera u odnocu na ulazni kanal
     *
     * @param inRow filter center row position over input
     * @param inCol filter center col position over input
     * @param ch filter output channel
     * @return
     */
    private float applyFilterInTheMiddleVectorized(final int inRow, final int inCol, final int ch, final float[] filterBuff, final float[] inputBuff) {
        float filterOut = 0;
        final Tensor3D inputs3d = (Tensor3D) inputs;

        // format za layout je channels first, row major, znaci CHW layout - tako je i u nasoj klasi Tensor
        // tf weights format: weights[kernel_height, kernel_width, kernel_depth, ch] 
        // ovo mozes da izbacis u konstantni atribut ili sl
        final int filterRowsColsDepth = filters.rows() * filters.cols() * filters.depth();
        final int filterChSize = filters.rows() * filters.cols();
        final int filterLength = filterWidth * filterHeight;
        final int inputRowsCols = inputs3d.rows() * inputs3d.cols();

        for (int inCh = 0; inCh < inputs3d.depth(); inCh++) { // iterate all input channels

            // kopiraj filter za inCh u filter buffer - filter debugovan dobro ga uzima
            final int srcPos = ch * filterRowsColsDepth + inCh * filterChSize;
            System.arraycopy(filters.getValues(),
                    srcPos, // odaklse ide src uzmi u obzir in inCh
                    filterBuff, // dest
                    0, // dest position
                    filterLength); //  filterBuff.length

            // kopiraj inpute koji se filtriraju u input buffer
            for (int filterRow = 0; filterRow < filterHeight; filterRow++) { // copy filtered input row by row            
                final int srcPos2 = inCh * inputRowsCols + (inRow - fCenterY + filterRow) * inputs3d.cols() + (inCol - fCenterX); // izmi u obzir filterRow
                System.arraycopy(inputs.getValues(), // src
                        srcPos2, // src pos
                        inputBuff, // destination
                        filterRow * filterWidth, // dest position
                        filterWidth); // length
            }

            // pomnozi sve elemente iz bufera i saberi - ovo bi trebalo da vektorizuje i eliminise getere
            for (int i = 0; i < filterLength; i++) {
                filterOut += inputBuff[i] * filterBuff[i];
            }
        } // inCh

        return filterOut;
    }

    private float applyFilterInTheMiddleVectorizedByRow(final int inRow, final int inCol, final int ch, final float[] filterBuff, final float[] inputBuff) {
        float filterOut = 0;

        final Tensor3D inputs3d = (Tensor3D) inputs;

        // format za layout je channels first, row major, znaci CHW layout - tako je i u nasoj klasi Tensor
        // tf weights format: weights[kernel_height, kernel_width, kernel_depth, ch] 
        // ovo mozes da izbacis u konstantni atribut ili sl
        final int filterRowsColsDepth = filters.rows() * filters.cols() * filters.depth();
        final int filterChSize = filters.rows() * filters.cols();

        for (int inCh = 0; inCh < inputs3d.depth(); inCh++) { // iterate all input channels
            int srcPos = ch * filterRowsColsDepth + inCh * filterChSize;
            for (int filterRow = 0; filterRow < filterHeight; filterRow++) { // copy filtered input row by row    
                // kopiraj filter za inCh u filter buffer - filter debugovan dobro ga uzima
                srcPos += filterRow * filterWidth;
                System.arraycopy(filters.getValues(),
                        srcPos, // odaklse ide src uzmi u obzir in inCh
                        filterBuff, // dest
                        0, // dest position
                        filterWidth); //  filterBuff.length

                // kopiraj inpute koji se filtriraju u input buffer
                final int srcPos2 = inCh * inputs3d.rows() * inputs3d.cols() + (inRow - fCenterY + filterRow) * inputs3d.cols() + (inCol - fCenterX); // izmi u obzir filterRow
                System.arraycopy(inputs.getValues(), // src
                        srcPos2, // src pos
                        inputBuff, // destination
                        0, // dest position
                        filterWidth); // length
            }

            // pomnozi sve elemente iz bufera i saberi - ovo bi trebalo da vektorizuje i eliminise getere
            for (int i = 0; i < filterWidth; i++) {
                filterOut += inputBuff[i] * filterBuff[i];
            }
        } // inCh

        return filterOut;
    }

    // ovo samo za ivice kad nije ceo filter - nije implementiran do kraja
    private float applyFilterOnEdgesVectorized(int inRow, int inCol, int ch) {
        float filterOut = 0;

        final Tensor3D inputs3d = (Tensor3D) inputs;

        // ovde imam jedan kanal filtera
        // format za layout je channels first, row major, znaci CHW layout - tako je i u nasoj klasi Tensor
        // tf weights format: weights[kernel_height, kernel_width, kernel_depth, ch] 
        final float[] filterBuff = new float[filterWidth * filterHeight]; // single filter channel -  razmisli da li mogu da uzmem i sve ulazne kanale
        final float[] inputBuff = new float[filterHeight * filterWidth]; // filtered input - ove buffere kreiraj kao atribute

        //if (inCol >= filterRadius && inCol < inputsCols-filterRadius-1 && 
        //                 inRow >= filterRadius && inRow < inputsRows-filterRadius-1)        
        float filterRadius = (filterWidth - 1) / 2;

        // dodaj jos i coskove/corners ili njih iskljuci i samo pune filtere po sirini i visini
        if (inRow < filterRadius && inCol >= filterRadius && inCol < inputs3d.cols() - filterRadius - 1) { // top edge
            final int srcPos3 = ch * filters.rows() * filters.cols() * filters.depth() + /* inCh *  */ filters.rows() * filters.cols(); // samo + filterWidth * rowsToSkip
            System.arraycopy(filters.getValues(),
                    srcPos3, // odaklse ide src uzmi u obzir in inCh
                    filterBuff, // dest - smanji filter buffer
                    0, // dest position
                    filterWidth * filterHeight); //  smani filter length             
        } else if (inRow > inputs3d.rows() - filterRadius - 1) { // bottom edge

        } else if (inCol < filterRadius) { // left edge

        } else if (inCol > inputs3d.cols() - filterRadius - 1) {

        }

        for (int inCh = 0; inCh < inputs3d.depth(); inCh++) { // iterate all input channels

            // kopiraj filter za inCh u filter buffer - filter debugovan dobro ga uzima
            final int srcPos = ch * filters.rows() * filters.cols() * filters.depth() + inCh * filters.rows() * filters.cols();
            System.arraycopy(filters.getValues(),
                    srcPos, // odaklse ide src uzmi u obzir in inCh
                    filterBuff, // dest
                    0, // dest position
                    filterWidth * filterHeight); //  filterBuff.length

            // kopiraj inpute koji se filtriraju u input buffer
            for (int filterRow = 0; filterRow < filterHeight; filterRow++) { // copy filtered input row by row            
                final int srcPos2 = inCh * inputs3d.rows() * inputs3d.cols() + (inRow - fCenterY + filterRow) * inputs3d.cols() + (inCol - fCenterX); // izmi u obzir filterRow
                System.arraycopy(inputs.getValues(), // src
                        srcPos2, // src pos
                        inputBuff, // destination
                        filterRow * filterWidth, // dest position
                        filterWidth); // length
            }

            // pomnozi sve elemente iz bufera i saberi - ovo bi trebalo da vektorizuje i eliminise getere
            for (int i = 0; i < filterBuff.length; i++) {
                filterOut += inputBuff[i] * filterBuff[i];
            }
        } // inCh

        return filterOut;
    }

    /**
     * Backward pass for convolutional layer tweaks the weights in filters.
     *
     * Next layer can be: FC, MaxPooling, Conv, (output same as FC), 1D or 3D
     * Prev layer can: Input, pool, conv, all 2D or 3D - all can be as
     * generalized 3D
     *
     * U 2 koraka:
     *
     * 1. povuci delte iz sledeceg lejera, i izracunaj tezinsku sumu delta za
     * sve neurone/outpute u ovom sloju 2. izracunaj promene tezina za sve veze
     * iz prethodnog lejera za svaki neuron/output u ovom sloju
     */
    @Override
    public void backward() {
        if (!isTrainable()) {
            return;
        }

        if (DeepNetts.getInstance().useCuda()) {
            backwardImpl.backward();        
        } else if (nextLayer instanceof FlattenLayer) {
            backwardFromFlatten();
        } else if (nextLayer instanceof MaxPoolingLayer) {
            backwardFromMaxPooling();           
        } else if (nextLayer instanceof ConvolutionalLayer) {
            backwardFromConvolutional();
        } else {
            throw new DeepNettsException("Backward propagation operation to convolutional layer not implemented for layer " + nextLayer.getClass());
        }
    }

    /**
     * Backward pass when next layer is flatten
     *
     * Calculates deltas for this layer
     *
     */
    private void backwardFromFlatten() {
        // ovaj se isto nece koristiti jer najcesce posle conv sloja ide maxpooling
        deltas.fill(0); // reset deltas for all units 

        // ali nemoj da ih stalno instanciras u svakom pasu nego negde u initu pre pocetka treninga
        if (!multithreaded) {
            if (batchMode) {
                //backwardFromFlattenForChannelBatch(0, depth);
                backwardImpl = new SingleThreadedBackwardBatchFromFlatten();
            } else {
                //backwardFromFlattenForChannel(0, depth);
                backwardImpl = new SingleThreadedBackwardFromFlatten();
            }

        } else {
            backwardImpl = new MultiThreadedBackwardFromFlatten();
        }

        backwardImpl.backward();
    }

    private class SingleThreadedBackwardFromFlatten implements Backward {

        @Override
        public void backward() {
            backwardFromFlattenForChannels(0, depth);
        }

    }

    private class SingleThreadedBackwardBatchFromFlatten implements Backward {

        @Override
        public void backward() {

            int fromChannel = 0, toChannel = depth;

            Tensor2D nextDeltas2D = (Tensor2D) nextLayer.deltas;
            Tensor4D deltas4D = (Tensor4D) deltas;
            Tensor4D outputs4D = (Tensor4D) outputs;
            for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
                for (int ch = fromChannel; ch < toChannel; ch++) {
                    for (int row = 0; row < deltas4D.rows(); row++) {
                        for (int col = 0; col < deltas4D.cols(); col++) { // ovaj povlaci delte direktno iz flatten layera
                            final int ndIdx = row * outputs4D.cols() * outputs4D.depth() + col * outputs4D.depth() + ch; //!!! ovaj nije dobar
                            final float nextDelta = nextDeltas2D.get(ndIdx, batchIdx); // u deltas u flatten layeru su zapravo weighted deltas propagorani iza fc layera vidi flatten bacward
                            final float afDerivative = activation.getPrime(outputs4D.get(batchIdx, ch, row, col));
                            deltas4D.set(nextDelta * afDerivative, batchIdx, ch, row, col); // set umesto add jer je vec sve sabrano u bacward od flatten - a batch??? trebalo bi da je to reseno u fc-u
                        }
                    }

                    // 2. calculate weight changes for this layer - ako je batch mode ona ne racunati nego samo akumulirati delte
                    calculateDeltaWeightsForChannelBatch(batchIdx, ch);
                }
            }
        }

    }

    private class MultiThreadedBackwardFromFlatten implements Backward {

        @Override
        public void backward() {
            try {
                threadPool.run(backwardFromFlattenTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage());
            }
        }

    }

    // ovaj nikad ne bi trebalo da se poziva jer je posle conv uvek maxpool a pr enjega flatten
    // trebalo bi ga popraviti jer je ovaj channels first
    private void backwardFromFlattenForChannels(final int fromChannel, final int toChannel) {
        // 1. Propagate deltas from the next flatten layer
        Tensor1D nextDeltas1D = (Tensor1D) nextLayer.deltas;
        Tensor3D deltas3D = (Tensor3D) deltas;
        Tensor3D outputs3D = (Tensor3D) outputs;
        for (int ch = fromChannel; ch < toChannel; ch++) {
            for (int row = 0; row < deltas3D.rows(); row++) {
                for (int col = 0; col < deltas3D.cols(); col++) { // ovaj povlaci delte direktno iz flatten layera
                    final int ndIdx = row * outputs3D.cols() * outputs3D.depth() + col * outputs3D.depth() + ch; //!!! ovaj nije dobar
                    final float nextDelta = nextDeltas1D.get(ndIdx); // u deltas u flatten layeru su zapravo weighted deltas propagorani iza fc layera vidi flatten bacward
                    final float afDerivative = activation.getPrime(outputs3D.get(ch, row, col));
                    deltas3D.set(nextDelta * afDerivative, ch, row, col); // set umesto add jer je vec sve sabrano u bacward od flatten - a batch??? trebalo bi da je to reseno u fc-u
                }
            }

            // 2. calculate weight changes for this layer - ako je batch mode ona ne racunati nego samo akumulirati delte
            calculateDeltaWeightsForChannel(ch);
        }
    }


    private void backwardFromMaxPooling() {
        final MaxPoolingLayer nextPoolLayer = (MaxPoolingLayer) nextLayer;
        maxIdx = nextPoolLayer.maxIdx; // uzmi index neurona koji je poslao max output na tekucu poziciju filtera

        deltas.fill(0); // reset all deltas

        if (!multithreaded) {
            if (!batchMode) {
                maxIdx = nextPoolLayer.maxIdx; // uzmi index neurona koji je poslao max output na tekucu poziciju filtera
                backwardFromMaxPoolingForChannelRange(0, this.depth);
            } else {
                maxIdxBatch = nextPoolLayer.maxIdxBatch;
                backwardFromMaxPoolingForChannelBatch(0, this.depth);
            }
            //}
        } else {
            try {
                threadPool.run(backwardFromPoolingTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage());
            }
        }
    }

    /**
     * Performs backward pass from maxpooling layer for specified channel in
     * this layer.
     *
     * @param ch
     */
    void backwardFromMaxPoolingForChannelRange(final int fromCh, final int toCh) {

        final Tensor3D outputs3d = (Tensor3D) outputs;
        final Tensor3D deltas3d = (Tensor3D) deltas;

        Tensor3D nextLayerDeltas3D = (Tensor3D) nextLayer.deltas;

        for (int ch = fromCh; ch < toCh; ch++) {
            // 1. Propagate deltas from next layer for max outputs from this layer
            for (int dr = 0; dr < nextLayerDeltas3D.rows(); dr++) { // sledeci lejer delte po visini
                for (int dc = 0; dc < nextLayerDeltas3D.cols(); dc++) { // sledeci lejer delte po sirini

                    final float nextLayerDelta = nextLayerDeltas3D.get(ch, dr, dc); // uzmi deltu iz sledeceg sloja za tekuci neuron sledeceg sloja
                    final int maxR = maxIdx[ch][dr][dc][0];
                    final int maxC = maxIdx[ch][dr][dc][1];

                    final float derivative = activation.getPrime(outputs3d.get(ch, maxR, maxC));
                    deltas3d.set(nextLayerDelta * derivative, ch, maxR, maxC);
                }
            } // end propagate deltas

            calculateDeltaWeightsForChannel(ch);
        }
    }

    void backwardFromMaxPoolingForChannelBatch(final int fromCh, final int toCh) {

        final Tensor4D outputs4d = (Tensor4D) outputs;
        final Tensor4D deltas4d = (Tensor4D) deltas;

        Tensor4D nextLayerDeltas4D = (Tensor4D) nextLayer.deltas;

        for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            for (int ch = fromCh; ch < toCh; ch++) {
                // 1. Propagate deltas from next layer for max outputs from this layer
                for (int dr = 0; dr < nextLayerDeltas4D.rows(); dr++) { // sledeci lejer delte po visini
                    for (int dc = 0; dc < nextLayerDeltas4D.cols(); dc++) { // sledeci lejer delte po sirini

                        final float nextLayerDelta = nextLayerDeltas4D.get(batchIdx, ch, dr, dc); // uzmi deltu iz sledeceg sloja za tekuci neuron sledeceg sloja
                        final int maxR = maxIdxBatch[batchIdx][ch][dr][dc][0];
                        final int maxC = maxIdxBatch[batchIdx][ch][dr][dc][1];

                        final float derivative = activation.getPrime(outputs4d.get(batchIdx, ch, maxR, maxC));
                        deltas4d.set(nextLayerDelta * derivative, batchIdx, ch, maxR, maxC);
                    }
                } // end propagate deltas

                calculateDeltaWeightsForChannelBatch(batchIdx, ch);
            }
        }
    }

    /**
     * Perform backpropagation from next convolutional layer.
     */
    private void backwardFromConvolutional() {
        deltas.fill(0); // reset all deltas in this layer (deltas are 3D)

        if (!multithreaded) {
            //for (int ch = 0; ch < this.depth; ch++) {
            backwardFromConvolutionalForChannel(0, this.depth);
            //}
        } else {
            try {
                threadPool.run(backwardFromConvolutionalTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage());
            }
        }

    }

    private void backwardFromConvolutionalForChannel(final int fromCh, final int toCh) { // fz == ch
        final Tensor3D outputs3d = (Tensor3D) outputs;
        final Tensor3D deltas3d = (Tensor3D) deltas;

        ConvolutionalLayer nextConvLayer = (ConvolutionalLayer) nextLayer;  // ovo u atribut i init metodu!!!
        final int filterCenterX = (nextConvLayer.filterWidth - 1) / 2;
        final int filterCenterY = (nextConvLayer.filterHeight - 1) / 2;

        Tensor3D nextLayerDeltas3D = (Tensor3D) nextLayer.deltas;

        for (int ch = fromCh; ch < toCh; ch++) {
            // 1. Propagate deltas from next conv layer for max outputs from this layer
            for (int ndZ = 0; ndZ < nextLayerDeltas3D.depth(); ndZ++) { // next conv layer depth / z / channel
                for (int ndRow = 0; ndRow < nextLayerDeltas3D.rows(); ndRow++) { // iteriraj delte sledeceg lejera po visini
                    for (int ndCol = 0; ndCol < nextLayerDeltas3D.cols(); ndCol++) { // iteriraj delte sledeceg lejera po sirini
                        final float nextLayerDelta = nextLayerDeltas3D.get(ndZ, ndRow, ndCol); // uzmi deltu iz sledeceg sloja za tekuci neuron (dx, dy, dz) sledeceg sloja, da li treba d ase sabiraju?

                        for (int fr = 0; fr < nextConvLayer.filterHeight; fr++) {
                            for (int fc = 0; fc < nextConvLayer.filterWidth; fc++) {
                                // ovo su input row and col from this layer - da li ovo dobro racuna - potvrdi
                                // stride je za koliko pomeri, ako pretpostavimo da je dobro
                                final int row = ndRow * nextConvLayer.stride + (fr - filterCenterY);
                                final int col = ndCol * nextConvLayer.stride + (fc - filterCenterX);

                                // ovo preskace on ekoje ispadaju
                                if (row < 0 || row >= outputs3d.rows() || col < 0 || col >= outputs3d.cols()) {
                                    continue;
                                }

                                // Mnoziti van petlje nakon zavrsetka sabiranja. Izracunati izvode u jednom prolazu, pa onda mnoziti  ane za svaku celiju.
                                final float afDerivative = activation.getPrime(outputs3d.get(ch, row, col)); // ne pozivati ovu funkciju ovde u petlji  vec optimizovati nekako. Mnoziti van petlje nakon zavrsetka sabiranja. Izracunati izvode u jednom prolazu, pa onda mnoziti  ane za svaku celiju.
                                // final float delta = nextLayerDelta * nextConvLayer.filters.get(fr, fc, ch, ndZ) * afDerivative; // pre okretanja redosleda parametara
                                final float delta = nextLayerDelta * nextConvLayer.filters.get(ndZ, ch, fr, fc) * afDerivative; // @TODO: check order of params                                
                                deltas3d.add(delta, ch, row, col); // da li su dobri row i col, razlike u ovom i sledecem lejeru nd
                            }
                        }
                        // bolje da ih ovde pomnozi za izodom af
                    }
                }
            }

            // 2. faza - izracunavanje promena tezina
            calculateDeltaWeightsForChannel(ch);
        }
    }

    /**
     * Calculates delta weights for the specified channel ch in this layer.
     *
     * @param ch channel/depth index
     */
    private void calculateDeltaWeightsForChannel(final int ch) {
        final Tensor3D inputs3d = (Tensor3D) inputs;
        final Tensor3D deltas3d = (Tensor3D) deltas;

        if (!batchMode) {
            Tensors.fillFourthDim(deltaWeights, ch, 0); // resetuje delta weights smo za filter ovog kanala da ne pregazi ono sto radi drugu threadovi
            deltaBiases.set(0, ch); // ovo moze da pregazi ako je multi threaded trebalo bi sve ovo setovati na nulu pre poziva ove metode
        }

        // 2. calculate weight changes in filters
        for (int deltaRow = 0; deltaRow < deltas3d.rows(); deltaRow++) {   // obidji sve celije u zadatom kanalu ch
            for (int deltaCol = 0; deltaCol < deltas3d.cols(); deltaCol++) {
                final int inRowStart = deltaRow * stride - fCenterY;
                final int inColStart = deltaCol * stride - fCenterX;

                // iterate all weights in filter
                for (int fz = 0; fz < filterDepth; fz++) { // filter depth, input channel
                    for (int fr = 0; fr < filterHeight; fr++) {
                        final int inRow = inRowStart + fr;
                        for (int fc = 0; fc < filterWidth; fc++) {

                            final int inCol = inColStart + fc;

                            if (inRow < 0 || inRow >= inputs3d.rows() || inCol < 0 || inCol >= inputs3d.cols()) {
                                continue; // ako je na ivici onda preskaci 
                            }
                            // fz je dubina filtera i to ide kroz sve ulazne kanale
                            final float input = inputs3d.get(fz, inRow, inCol); // get input for this output and weight

                            float regularization = 0;
                            if (regL2 != 0 && regL1 != 0) {
                                regularization = (regL2 != 0
                                        ? regL2 * 2 * weights.get(ch, fz, fr, fc)
                                        : regL1 * MathFunctions.absPrime(weights.get(ch, fz, fr, fc)));
                            }

                            // regularizacija je deo gradijenta, i racuna se za svaki ulaz/tezinu se racuna sto znaci da sve to zajedno treba uproseciti zbog shared weights
                            final float grad = deltas3d.get(ch, deltaRow, deltaCol) * input + regularization; // ovo bi trebalo ovde a ne ispod delta / divisor 
                            // u batch modu mozda mogu samo da akumuliram delte pa onda na kraju da izracunam deltaWeight
                            gradients.add(grad, ch, fz, fr, fc);
                            final float deltaWeight = optimizer.calculateDeltaWeight(grad, ch, fz, fr, fc);
                            deltaWeights.add(deltaWeight, ch, fz, fr, fc);
                        }
                    }
                }
                float deltaBias = optimizer.calculateDeltaBias(deltas3d.get(ch, deltaRow, deltaCol), ch);
                deltaBiases.add(deltaBias, ch);                
            } // end for deltaCols
        } // end for delta rows / calculate weight changes in filter   
    }

    private void calculateDeltaWeightsForChannelBatch(final int batchIdx, final int ch) {
        final Tensor4D inputs4d = (Tensor4D) inputs;
        final Tensor4D deltas4d = (Tensor4D) deltas;

        // todo - ovo integristti u racunanje delte da ne iteriram ceo kanal dva puta bez potrebe
        if (!batchMode) {
            Tensors.fillFourthDim(deltaWeights, ch, 0); // resetuje delta weights smo za filter ovog kanala da ne pregazi ono sto radi drugu threadovi
            deltaBiases.set(0, ch); // ovo moze da pregazi ako je multi threaded trebalo bi sve ovo setovati na nulu pre poziva ove metode
        }

        // 2. calculate weight changes in filters
        for (int deltaRow = 0; deltaRow < deltas4d.rows(); deltaRow++) {   // obidji sve celije u zadatom kanalu ch
            for (int deltaCol = 0; deltaCol < deltas4d.cols(); deltaCol++) {
                final int inRowStart = deltaRow * stride - fCenterY;
                final int inColStart = deltaCol * stride - fCenterX;

                // iterate all weights in filter
                for (int fz = 0; fz < filterDepth; fz++) { // filter depth, input channel
                    for (int fr = 0; fr < filterHeight; fr++) {
                        final int inRow = inRowStart + fr;
                        for (int fc = 0; fc < filterWidth; fc++) {

//                            final int inRow = deltaRow * stride + fr - fCenterY;
//                            final int inCol = deltaCol * stride + fc - fCenterX;
//                            final int inRow = inRowStart + fr;
                            final int inCol = inColStart + fc;

                            if (inRow < 0 || inRow >= inputs4d.rows() || inCol < 0 || inCol >= inputs4d.cols()) {
                                continue; // ako je na ivici onda preskaci 
                            }
                            // fz je dubina filtera i to ide kroz sve ulazne kanale
                            final float input = inputs4d.get(batchIdx, fz, inRow, inCol); // get input for this output and weight

                            float regularization = 0;
                            if (regL2 != 0 && regL1 != 0) {
                                regularization = (regL2 != 0
                                        ? regL2 * 2 * weights.get(ch, fz, fr, fc)
                                        : regL1 * MathFunctions.absPrime(weights.get(ch, fz, fr, fc)));
                            }
                            // ako je na ivici onda ne treba da propagira na sve mozda zbog toga buguje na ivicama...   neke tezne/delte ne treba da delim                        

                            // regularizacija je deo gradijenta, i racuna se za svaki ulaz/tezinu se racuna sto znaci da sve to zajedno treba uproseciti zbog shared weights
                            final float grad = deltas4d.get(batchIdx, ch, deltaRow, deltaCol) * input + regularization; // ovo bi trebalo ovde a ne ispod delta / divisor 
                            gradients.add(grad, ch, fz, fr, fc);
                            // u batch modu mozda mogu samo da akumuliram delte pa onda na kraju da izracunam deltaWeight
                            final float deltaWeight = optimizer.calculateDeltaWeight(grad, ch, fz, fr, fc);
                            deltaWeights.add(deltaWeight, ch, fz, fr, fc);
                        }
                    }
                }
                float deltaBias = optimizer.calculateDeltaBias(deltas4d.get(batchIdx, ch, deltaRow, deltaCol), ch);
                //deltaBias /= chAvgDivisor; // zato sto je bias usao u ukupnu sumu na svokoj poziciji kanala
                deltaBiases.add(deltaBias, ch); // ovo je pitanje z abatch, da li raditi mean negde
            }
            // verovatno bih mogao i ovde sve da ih uprosecim za sve pozicije...
        } // end calculate weight changes in filter
    }

    /**
     * Apply weight changes calculated in backward pass
     */
    @Override
    public void applyWeightChanges() {
        if (!isTrainable()) {
            return; // if layer is not trainable do not apply any changes
        }
//        if (batchMode) {    // divide biases with batch samples if it is in batch mode
//            deltaWeights.div(batchSize);
//            deltaBiases.div(batchSize);
//        }

        // ovo samo ako je ukljucen i momentum
        Tensors.copy(deltaWeights, prevDeltaWeights); // da li ovo treba pre ilo posle prethodnog kad aje u batch mode-u?, ok je d abude posle jer se prienjuje pojedinacno
        Tensors.copy(deltaBiases, prevDeltaBiases);  // save this for momentum

        // apply weight and bias changes
        if (!DeepNetts.getInstance().useCuda()) {        
            filters.add(deltaWeights);
            biases.add(deltaBiases);
        }

        if (batchMode) {    // reset delta weights for next batch
            deltaWeights.fill(0);
        }

        if (batchMode) { // reset delta biases for next batch
            deltaBiases.fill(0);
        }

        if (DeepNetts.getInstance().useCuda()) {
//            weights.copyToGPU(); // sad se sabiraju na gpu
//            biases.copyToGPU();
//            weights.copyFromGPU();
//            biases.copyFromGPU();
         }
        
    }

    public TensorBase getFilters() {
        return filters;
    }

    @Override
    public Tensor4D getWeights() {
        return filters;
    }

    @Override
    public void setWeights(Tensor4D weights) {
        setFilters(weights);
    }

    public void setFilters(Tensor4D filters) {
        this.filters = filters;
        this.weights = filters;
    }

    public void setFilters(String filtersStr) {
        int filterSize = filterWidth * filterHeight * filterDepth;

        float[] filterValues = new float[filterSize];
        String[] vals = filtersStr.split(",");
        for (int k = 0; k < filterSize; k++) {
            filterValues[k] = Float.parseFloat(vals[k]);
        }

        filters.setValues(filterValues); // ovde je tensor 5x5x3 a imamomo samo 25 vrednosti

    }

    public int getFilterWidth() {
        return filterWidth;
    }

    public int getFilterHeight() {
        return filterHeight;
    }

    public int getFilterDepth() {
        return filterDepth;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }

    public TensorBase getFilterDeltaWeights() {
        return deltaWeights;
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

    @Override
    public float getL1WeightSum() {
        return filters.sumAbs();
    }

    @Override
    public float getL2WeightSum() {
        return filters.sumSqr();
    }

    //public void init()
    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
        ois.defaultReadObject();
    }

    @Override
    public String toString() {
        return "Convolutional Layer { filter width:" + filterWidth + ", filter height: " + filterHeight + ", channels: " + depth + ", stride: " + stride + ", activation: " + activationType.name() + "}";
    }

}
