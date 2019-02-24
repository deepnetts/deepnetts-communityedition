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

package deepnetts.net.layers;

import deepnetts.net.layers.activation.ActivationFunctions;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.core.DeepNetts;
import deepnetts.net.train.opt.Optimizers;
import deepnetts.util.DeepNettsException;
import deepnetts.util.WeightsInit;
import deepnetts.util.Tensor;
import java.util.logging.Logger;
import deepnetts.net.layers.activation.ActivationFunction;

/**
 * Convolutional layer performs image convolution operation on outputs of a
 * previous layer using filters. This filtering operation is similar like applying
 * image filters in photoshop, but this filters can also be trained to learn
 * image features of interest.
 *
 * Layer include parameters:
 * filter's width, heigh
 * Number of filters / depth
 * Step when applying filters : stride
 * Padding, which is an image border to keep the size of image and avoid information loss
 * padding Stride defaults to 1
 *
 * @author Zoran Sevarac
 */
public class ConvolutionalLayer extends AbstractLayer {

    Tensor[] filters;           // each filter corresponds to a single channel. Each filter can be 3D, where 3rd dimension coreesponds to depth in previous layer. TODO: the depth pf th efilter should be tunable
    Tensor[] deltaWeights;      //ovo za sada ovako dok proradi. Posle mozda ubaciti jos jednu dimenziju u matricu - niz za kanale. i treba da overriduje polje jer su weights u filterima za sve prethdne kanale

    /**
     * Convolutional filter width
     */
    int filterWidth,

    /**
     * Filter dimensions, filter depth is equal to number of depth / depth of
     */
    filterHeight,

    /**
     * Filter dimensions, filter channels is equal to number of channels / channels of
     */

    /**
     * Filter dimensions, filter depth is equal to number of depth / depth of
     */
    filterDepth; // da li je filter istih dimenzija za sve feature mape?

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


    private static Logger LOG = Logger.getLogger(DeepNetts.class.getName());

    /**
     * Create a new instance of convolutional layer with specified filter.
     * dimensions, default padding (filter-1)/2, default stride stride value 1, and specified number of channels.
     *
     * @param filterWidth
     * @param filterHeight
     * @param channels
     */
    public ConvolutionalLayer(int filterWidth, int filterHeight, int channels) {
        // sve mora da bude pozitivno. filteri motaju da budu  neparni - validacija

        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.depth = channels; // ovo je isto kao i depth, broj feature mapa
        this.stride = 1;
        this.activationType = ActivationType.TANH; // use relu as default?
        this.activation = ActivationFunction.create(activationType);
    }

    public ConvolutionalLayer(int filterWidth, int filterHeight, int channels, ActivationType activationType) {
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.depth = channels;
        this.stride = 1;
        this.activationType = activationType;
        this.activation = ActivationFunction.create(activationType);
    }

    public ConvolutionalLayer(int filterWidth, int filterHeight, int stride, int channels, ActivationType activationType) {
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.depth = channels;
        this.stride = stride;
        this.activationType = activationType;
        this.activation = ActivationFunction.create(activationType);
    }


    /**
     * Init dimensions, create matrices, filters, weights, biases and all
     * internal structures etc.
     *
     * Assumes that prevLayer is set in network builder
     */
    @Override
    public void init() {
        // prev layer can only be input, max pooling or convolutional
        if (!(prevLayer instanceof InputLayer || prevLayer instanceof ConvolutionalLayer || prevLayer instanceof MaxPoolingLayer)) {
            throw new DeepNettsException("Illegal architecture: convolutional layer can be used only after input, convolutional or maxpooling layer");
        }

        inputs = prevLayer.outputs;

        width = (prevLayer.getWidth()) / stride;
        height = (prevLayer.getHeight()) / stride;
        // depth is set in constructor

        fCenterX = (filterWidth-1) / 2; //  padding = filter /2
        fCenterY = (filterHeight-1) / 2;

        // init output cells, deltas and derivative buffer
        outputs = new Tensor(height, width,  depth);
        deltas = new Tensor(height, width, depth);
//        derivatives = new Tensor(height, width, depth);

        // init filters(weights) - broj filtera je isti kao i broj kanala/dubina prethodnog lejera
        filterDepth = prevLayer.getDepth();
        filters = new Tensor[depth]; // depth of the filters should be configurable - its a hyper param!
        deltaWeights = new Tensor[depth];

        int inputCount = (filterWidth * filterHeight + 1) * filterDepth;

        // kreiraj pojedinacne filtere ovde
        for (int ch = 0; ch < filters.length; ch++) {
            filters[ch] = new Tensor(filterHeight, filterWidth, filterDepth);
            WeightsInit.uniform(filters[ch].getValues(), inputCount); // vidi koji algoritam da koristim ovde: uzmi u obzir broj kanala i dimenzije filtera pa da im suma bude 1 ili sl. neka gausova distribucija...

            deltaWeights[ch] = new Tensor(filterHeight, filterWidth, filterDepth);
        }

        // and biases               // svaki kanal ima svoj filter i svoj bias - sta ako prethodni sloj ima vise biasa? mislim da bi tada svaki filter trebalo da ima svoj bias ovo bi znaci trebalo da bude 2D biases[depth][prevLayerDepth]
        biases = new float[depth]; // biasa ima koliko ima kanala - svaki FM ima jedan bias - mozda i vise... ili ce svaki filter imati svoj bias? - tako bi trebalo...
                                    // svaki kanal u ovom sloju ima filtera onoliko kolik ima kanala u prethodnom sloji. i Svi ti filteri imaju jedan bias
        deltaBiases = new float[depth];
  //      WeightsInit.randomize(biases);        // sometimes the init to 0
    }

    /**
     * Forward pass for convolutional layer.
     * Performs convolution operation on output from previous layer using filters in this layer, on all channels.
     * Each channel from prev layer has its own filter (3D filter), and every channel in this layer has its 3D filter used to scan all channels in prev layer.
     *
     * Previous layers can be: Input, MaxPooling or Convolutional.
     *
     * For more about convolution see http://www.songho.ca/dsp/convolution/convolution.html
     */
    @Override
    public void forward() {

        // paralelieze this external loop - channels
        for (int ch = 0; ch < this.depth; ch++) {
            int outR = 0, outC = 0; // reset indexes for current output's row and col

            for (int inR = 0; inR < inputs.getRows(); inR += stride) { // iterate all input rows
                outC = 0; // every time when input goes in next row, output does too, so reset column idx

                for (int inC = 0; inC < inputs.getCols(); inC += stride) { // iterate all input cols
                   outputs.set(outR, outC, ch, biases[ch]); // sum will be added to bias - I can set entire matrix to bias initial values  above

                    // apply filter to all channnels in previous layer
                    for (int fz = 0; fz < filterDepth; fz++) { // iterate filter by depth - all input channels (in previous layer)
                        for (int fr = 0; fr < filterHeight; fr++) { // iterate filter by height/rows
                            for (int fc = 0; fc < filterWidth; fc++) { // iterate filter by width / columns
                                final int cr = inR + (fr - fCenterY); // convolved row idx
                                final int cc = inC + (fc - fCenterX); // convolved col idx

                                // skip input indexes which are out of bounds
                                if (cr < 0 || cr >= inputs.getRows() || cc < 0 || cc >= inputs.getCols()) continue;

                                final float out = inputs.get(cr, cc, fz) * filters[ch].get(fr, fc, fz); // output of a single conv filter cell
                                outputs.add(outR, outC, ch, out); // accumulate filters from all channels
                            }
                        }
                    }

                    // apply activation function
                    final float out = ActivationFunctions.calc(activationType, outputs.get(outR, outC, ch));
                    outputs.set(outR, outC, ch, out);
                    outC++; // move to next col in out layer after each filter position
                }
                outR++; // every time input goes to next row, output does too
            }
        }
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
     * sve neurone/outpute u ovom sloju
     * 2. izracunaj promene tezina za sve veze
     * iz prethodnog lejera za svaki neuron/output u ovom sloju
     */
    @Override
    public void backward() {
        if (nextLayer instanceof FullyConnectedLayer) {
            backwardFromFullyConnected();
        }

        if (nextLayer instanceof MaxPoolingLayer) {
            backwardFromMaxPooling();
        }

        if (nextLayer instanceof ConvolutionalLayer) {
            // NOTE: average weights for the filter and biases? - koliko ima pozicija i kanala??? negde sam procitao da treba da se sabiraju...
            backwardFromConvolutional();
        }
    }

    /**
     * Backward pass when next layer is fully connected.
     *
     * Calculates deltas for this layer
     *
     */
    private void backwardFromFullyConnected() {
        deltas.fill(0); // reset deltas for all units

        // todo : Ako je svaki sa svakim povezanost mozda ovo moze da se ovo odradi u manje petlji - tipa samo dve
        for (int ch = 0; ch < this.depth; ch++) { // iteriraj sve kanale/feature mape u ovom lejeru - razbij kanale sa fork join frejmvorkom

            // 1. Propagate deltas from the next FC layer
            for (int row = 0; row < this.height; row++) {
                for (int col = 0; col < this.width; col++) {
                    final float derivative = ActivationFunctions.prime(activationType, outputs.get(row, col, ch)); // dy/ds
                    for (int ndC = 0; ndC < nextLayer.deltas.getCols(); ndC++) { // sledeci lejer delte po sirini/kolone posto je fully connected
                        final float delta = nextLayer.deltas.get(ndC) * nextLayer.weights.get(col, row, ch, ndC) * derivative; // TODO: mnoziti sumu na kraju samo jednom optimizacija -mozda
                        deltas.add(row, col, ch, delta);
                    }
                }
            } // end back propagate deltas
            // ovd bi moglo da se mnozi sa derivative ovde da bi smanjio broj operacija?
            // onda bi i ovo ispod moralo van ove petlje - ali ako je pparalelizuje po kanalima onda je ok

            // 2. calculate weight changes for this layer
            calculateDeltaWeights(ch); // kanali se mogu paralelizovati
        } // end channel iterator
    }

    private void backwardFromMaxPooling() {
        final MaxPoolingLayer nextPoolLayer = (MaxPoolingLayer) nextLayer;
        final int[][][][] maxIdx = nextPoolLayer.maxIdx; // uzmi index neurona koji je poslao max output na tekucu poziciju filtera

        deltas.fill(0); // reset all deltas

        for (int ch = 0; ch < this.depth; ch++) {  // iteriraj sve kanale u ovom lejeru (to su automatski i kanali u sledem max pooling lejeru)
            // 1. Propagate deltas from next layer for max outputs from this layer
            for (int dr = 0; dr < nextLayer.deltas.getRows(); dr++) { // sledeci lejer delte po visini
                for (int dc = 0; dc < nextLayer.deltas.getCols(); dc++) { // sledeci lejer delte po sirini

                    final float nextLayerDelta = nextLayer.deltas.get(dr, dc, ch); // uzmi deltu iz sledeceg sloja za tekuci neuron sledeceg sloja
                    final int maxR = maxIdx[ch][dr][dc][0];
                    final int maxC = maxIdx[ch][dr][dc][1];

                    final float derivative = ActivationFunctions.prime(activationType, outputs.get(maxR, maxC, ch));
                    deltas.set(maxR, maxC, ch, nextLayerDelta * derivative);
                }
            } // end propagate deltas

            calculateDeltaWeights(ch);
        } // end channel iterator
    }

    private void backwardFromConvolutional() {
        ConvolutionalLayer nextConvLayer = (ConvolutionalLayer) nextLayer;
        deltas.fill(0); // reset all deltas in this layer (deltas are 3D)
        int filterCenterX = (nextConvLayer.filterWidth - 1) / 2;
        int filterCenterY = (nextConvLayer.filterHeight - 1) / 2;

        /// !!! Zasto u forward pasu imam 6 a u backward 7 loopova? - ako imam gz i ndZ da li mi treba ch ???
       // for (int ch = 0; ch < this.depth; ch++) {  // iteriraj sve kanale/feature mape u ovom lejeru
            // 1. Propagate deltas from next conv layer for max outputs from this layer
            for (int ndZ = 0; ndZ < nextLayer.deltas.getDepth(); ndZ++) { // iteriraj sve kanale sledeceg sloja - ovde se moze paralelizovati takodje!
                for (int ndRow = 0; ndRow < nextLayer.deltas.getRows(); ndRow++) { // iteriraj delte sledeceg lejera po visini
                    for (int ndCol = 0; ndCol < nextLayer.deltas.getCols(); ndCol++) { // iteriraj delte sledeceg lejera po sirini
                        final float nextLayerDelta = nextLayer.deltas.get(ndRow, ndCol, ndZ); // uzmi deltu iz sledeceg sloja za tekuci neuron (dx, dy, dz) sledeceg sloja, da li treba d ase sabiraju?

                        for (int fz = 0; fz < nextConvLayer.filterDepth; fz++) {    //!! pa da li je ovo isto kao i prvi loop - da li je dupliranje???!!! kao ovaj prvi ch!!! mislim da tu ima nepotrebnog preklapanja /dupliranja. kada filter ne bude iseo preko svih u prethodnom lejeru onda nece biti preklapanja
                            for (int fr = 0; fr < nextConvLayer.filterHeight; fr++) {
                                for (int fc = 0; fc < nextConvLayer.filterWidth; fc++) {
                                    final int row = ndRow * nextConvLayer.stride + (fr - filterCenterY); // proveri da li ovo dobro racuna
                                    final int col = ndCol * nextConvLayer.stride + (fc - filterCenterX);

                                    if (row < 0 || row >= outputs.getRows() || col < 0 || col >= outputs.getCols()) continue;

                                    final float derivative = ActivationFunctions.prime(activationType, outputs.get(row, col, fz)); // ne pozivati ovu funkciju ovde u petlji  vec optimizovati nekako. Mnoziti van petlje nakon zavrsetka sabiranja. Izracunati izvode u jednom prolazu, pa onda mnoziti  ane za svaku celiju.
                                    //   ... ovde treba razjasniti kako se mnozi sa weightsomm? da li ih treba sabirati
                                    deltas.add(row, col, fz, nextLayerDelta * nextConvLayer.filters[ndZ].get(fr, fc, fz) * derivative);
                                }
                            }
                        }
                    }
                }
            }

            for (int ch = 0; ch < this.depth; ch++) {
            // 2. calculate delta weights for this layer (this is same for all types of next layers)
                calculateDeltaWeights(ch); // split these into fork joins too
            }
    }

    /**
     * Calculates delta weights for the specified channel ch in this convolutional layer.
     *
     * @param ch channel/depth
     */
    private void calculateDeltaWeights(int ch) {
        if (!batchMode) {
            deltaWeights[ch].fill(0); // reset all delta weights for the current channel - these are 4d matrices
            deltaBiases[ch] = 0; // da li b ovo trebalo da bude 2d niz? verovatno ne
        }

        final float divisor = width * height; //  tezine u filteru racunari kao prosek sa svih pozicija u feature mapi

        // assumes that deltas from the next layer are allready propagated

        // 2. calculate weight changes in filters
        for (int deltaRow = 0; deltaRow < deltas.getRows(); deltaRow++) {
            for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) {
                // iterate all weights in filter for filter depth
                for (int fz = 0; fz < filterDepth; fz++) { // filter depth, input channel
                    for (int fr = 0; fr < filterHeight; fr++) {
                        for (int fc = 0; fc < filterWidth; fc++) {

                            final int inRow = deltaRow * stride + fr - fCenterY;
                            final int inCol = deltaCol * stride + fc - fCenterX;

                            if (inRow < 0 || inRow >= inputs.getRows() || inCol < 0 || inCol >= inputs.getCols()) continue;

                            final float input = inputs.get(inRow, inCol, fz); // get input for this output and weight; padding?
                            final float grad = deltas.get(deltaRow, deltaCol, ch) * input;

                            float deltaWeight = 0;
                            switch (optimizer) {
                                case SGD:
                                    deltaWeight = Optimizers.sgd(learningRate, grad);
                                    break;
                            }
                            deltaWeight /=divisor;
                            deltaWeights[ch].add(fr, fc, fz, deltaWeight);
                        }
                    }
                }
               float deltaBias=0;
                switch (optimizer) {
                    case SGD:
                        deltaBias = Optimizers.sgd(learningRate, deltas.get(deltaRow, deltaCol, ch));
                        break;
                }
                deltaBiases[ch] /=divisor;
                deltaBiases[ch] += deltaBias;
            }
        } // end calculate weight changes in filter   }
    }

    /**
     * Apply weight changes calculated in backward pass
     */
    @Override
    public void applyWeightChanges() {

        if (batchMode) {
            Tensor.div(deltaBiases, batchSize);
        }

        for (int ch = 0; ch < depth; ch++) {
            if (batchMode) {
                deltaWeights[ch].div(batchSize);
            }

            filters[ch].add(deltaWeights[ch]);
            biases[ch] += deltaBiases[ch];

            if (batchMode) {    // reset delta weights for next batch
                deltaWeights[ch].fill(0);
            }
        }

        if (batchMode) { // reset delta biases for next batch
            Tensor.fill(deltaBiases, 0);
        }

    }

    public Tensor[] getFilters() {
        return filters;
    }

    public void setFilters(Tensor[] filters) {
        this.filters = filters;
    }

    public void setFilters(String filtersStr) {

        String[] strVals = filtersStr.split(";"); // ; is hardcoded filter separator see FileIO // also can be splited at "
        int filterSize = filterWidth * filterHeight * filterDepth;

        for (int i = 0; i < filters.length; i++) {
            float[] filterValues = new float[filterSize];
            String[] vals = strVals[i].split(",");
            for (int k = 0; k < filterSize; k++) {
                filterValues[k] = Float.parseFloat(vals[k]);
            }

            filters[i].setValues(filterValues); // ovde je tensor 5x5x3 a imamomo samo 25 vrednosti
        }
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

    public Tensor[] getFilterDeltaWeights() {
        return deltaWeights;
    }

}
