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

import deepnetts.util.Tensor;

/**
 * This layer performs max pooling operation in convolutional neural network, which
 * scales down output from previous layer by taking max outputs from small predefined filter areas.
 *
 * @see ConvolutionalNetwork
 * @author Zoran Sevarac
 */
public class MaxPoolingLayer extends AbstractLayer {

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
     * Remember idx of max output for each filter position. [channel][row][col][2]
     */
    int maxIdx[][][][];


    /**
     * Creates a new max pooling layer with specified filter dimensions and stride.
     *
     * @param filterWidth width of the filter square
     * @param filterHeight height of the filter square
     * @param stride filter step
     */
    public MaxPoolingLayer(int filterWidth, int filterHeight, int stride) {
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.stride = stride;
    }

    @Override
    final public void init() {
        // max pooling layer can be only after Convolutional Layer
        if (!(prevLayer instanceof ConvolutionalLayer)) throw new RuntimeException("Illegal network architecture! MaxPooling can be only after convolutional layer!");

        inputs = prevLayer.outputs;

        width = (inputs.getCols() - filterWidth) / stride + 1; // ovo mora biti ceo broj strude veci od 2, 3 je suvise destruktivan
        height = (inputs.getRows() - filterHeight) / stride + 1;
        depth = prevLayer.getDepth(); // depth of pooling layer is always same as in previous convolutional layer

        outputs = new Tensor(height, width, depth);
        deltas = new Tensor(height, width,  depth);

        // used in fprop to save idx position of max value
        maxIdx = new int[depth][height][width][2]; // svakoj poziciji filtera odgovara jedna [row, col] celija u outputu idx 0 je col, idx 1 je row
    }


    /**
     * Max pooling forward pass outputs the max value for each filter position.
     */
    @Override
    public void forward() {
        float max; // max value
        int maxC = -1, maxR = -1;

        for (int ch = 0; ch < this.depth; ch++) {  // iteriraj sve kanale/feature mape u ovom lejeru
            int outCol = 0, outRow = 0;

            for (int inRow = 0; inRow < inputs.getRows() - filterHeight + 1; inRow += stride) {
                outCol = 0; // reset col on every new row ???????
                for (int inCol = 0; inCol < inputs.getCols() - filterWidth + 1; inCol += stride) {

                    // apply max pool filter
                    max = inputs.get(inRow, inCol, ch);
                    maxC = inCol;
                    maxR = inRow;
                    for (int fr = 0; fr < filterHeight; fr++) {
                        for (int fc = 0; fc < filterWidth; fc++) {
                            if (max < inputs.get(inRow + fr, inCol + fc, ch)) {
                                maxR = inRow + fr;
                                maxC = inCol + fc;
                                max = inputs.get(maxR, maxC, ch);
                            }
                        }
                    }

                    // zapamti indexe neurona iz prethodnog lejera koji su propustili max (koristice se u bacward pass-u)
                    maxIdx[ch][outRow][outCol][0] = maxR; // height idx (row)
                    maxIdx[ch][outRow][outCol][1] = maxC; // width idx (col)

                    outputs.set(outRow, outCol, ch, max); // set max value as output
                    outCol++;   // increase output col by one for each input (stride) step
                } // scan col
                outRow++; // increase output row by one for each input (stride) step
            } // scan row
        } // channel/depth
    }

    /**
     * backward pass for a max(x, y) operation has a simple interpretation as
     * only routing the gradient to the input that had the highest value in the
     * forward pass. Hence, during the forward pass of a pooling layer it is
     * common to keep track of the index of the max activation (sometimes also
     * called the switches) so that gradient routing is efficient during
     * backpropagation.
     *
     * backward error pass samo kroz index oji je prosao forward pass
     *
     */
    @Override
    public void backward() {
        // propusti gresku iz sledeceg lejera daltu unazad samo za neurone koji su bili max (na osnovu zapamcenih pozicija)
        // kako su povezani max pooling i sledeci conv layer?  standardna konvolucija
        // prvo treba propagirati delte iz narednog lejera u ovaj lejer
        // zapravo ovde treba samo preneti weighted deltas unazad a u prethodnom konvolucionom sloju se vrsi selekcija u skladu sa max ulazom itd.

        if (nextLayer instanceof FullyConnectedLayer) {
            backwardFromFullyConnected();
        }

        else if (nextLayer instanceof ConvolutionalLayer) {
            // iterate all deltas  in next layer
            final ConvolutionalLayer nextConvLayer = (ConvolutionalLayer)nextLayer;
            deltas.fill(0);
            final int filterCenterX = (nextConvLayer.filterWidth-1) / 2;
            final int filterCenterY = (nextConvLayer.filterHeight-1) / 2;

         //  for (int ch = 0; ch < this.depth; ch++) {  // iteriraj sve kanale/feature mape u ovom lejeru, odnosno odgovarajuce filtere u sledecem // umesto ovog ici dole na fz
                // 1. Propagate deltas from next conv layer for max outputs from this layer
                for (int ndz = 0; ndz < nextLayer.deltas.getDepth(); ndz++) { // iteriraj i 3-cu dimeziju sledeceg sloja
                    for (int ndr = 0; ndr < nextLayer.deltas.getRows(); ndr++) { // sledeci lejer delte po visini
                        for (int ndc = 0; ndc < nextLayer.deltas.getCols(); ndc++) { // sledeci lejer delte po sirini
                            final float nextLayerDelta = nextLayer.deltas.get(ndr, ndc, ndz); // uzmi deltu iz sledeceg sloja za tekuci neuron (dx, dy, dz) sledeceg sloja

                            for (int fz = 0; fz < nextConvLayer.filterDepth; fz++) {
                                for (int fr = 0; fr < nextConvLayer.filterHeight; fr++) {
                                    for (int fc = 0; fc < nextConvLayer.filterWidth; fc++) {
                                        final int outRow = ndr * nextConvLayer.stride + (fr - filterCenterY);
                                        final int outCol = ndc * nextConvLayer.stride + (fc - filterCenterX);

                                        if (outRow < 0 || outRow >= outputs.getRows() || outCol < 0 || outCol >= outputs.getCols()) continue;

                                        // svaki filter propagira unazad svoju deltu, ne bi trebalo mesati delte iz razlicith kanala/filtera vec pre srednja vrednost ili sl?
                                        deltas.add(outRow, outCol, fz, nextLayerDelta * nextConvLayer.filters[ndz].get(fr, fc, fz));
                                                        /// ovde sam umesto ch stavio fz a gore ch iskomentarisao!!! tako treba, jos potvrdi
                                    }
                                }
                            }
                        }
                    }
                }
           // }
        }

        // we can also put zeros to all deltas that dont bellong to max outputs, and free prev convolutional layer to do that...
    }

    private void backwardFromFullyConnected() {
        deltas.fill(0);

        for (int ch = 0; ch < deltas.getDepth(); ch++) {  // iteriraj sve kanale/feature mape u ovom lejeru
            for (int row = 0; row < deltas.getRows(); row++) {
                for (int col = 0; col < deltas.getCols(); col++) {
                    for (int ndC = 0; ndC < nextLayer.deltas.getCols(); ndC++) { // sledeci lejer iteriraj delte po sirini/kolonama posto je fully connected
                        final float nextDelta = nextLayer.deltas.get(ndC);
                        final float weight = nextLayer.weights.get(col, row, ch, ndC);
                        deltas.add(row, col, ch, nextDelta * weight);
                    }
                }
            }
        }
    }


    /**
     * Does nothing for pooling layer since it does not have weights
     * It just propagates deltas from next layer to previous through connections that had max activation in forward pass
     */
    @Override
    public void applyWeightChanges() {    }

    public int getFilterWidth() {
        return filterWidth;
    }

    public int getFilterHeight() {
        return filterHeight;
    }

    public int getStride() {
        return stride;
    }






}