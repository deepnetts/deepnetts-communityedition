package deepnetts.util;

import deepnetts.tensor.Shape;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.TensorBase;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor4D;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.ConvolutionalLayer;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class TensorflowUtils {
    

    public static void importWeights(NeuralNetwork network, String weightsFileName) throws IOException {                        
        StringBuffer weightsValuesBuf = new StringBuffer();
        boolean readingWeightLines = false;

        int currLayerIdx = -1;
        AbstractLayer currLayer = null;
        TensorBase weightsTensor = null;
        
        //nemoj odjednom ovo da ucitavas nego red po red jer je problem ako je mreza prevelika        
        // List<String> lines = Files.readAllLines(Paths.get(weightsFileName));
        File weightsFile = new File(weightsFileName);
        BufferedReader br = new BufferedReader(new FileReader(weightsFile));
        String line = null;
        boolean exportedInputLayer = false;
        while ( (line = br.readLine()) != null) { 
                                                        
            if (line.isEmpty()) continue; // skip empty lines
            
            // check if input layer is exprted
            if (line.contains("\"name\": \"input")) {
                exportedInputLayer = true;
                continue;
            } // skip input layer
          
            if (line.startsWith("layer:")) { // found next layer 
                String layerIdxStr = line.substring(line.indexOf(":") + 1); // get layer index
                currLayerIdx = Integer.parseInt(layerIdxStr); // -1 jer exportovani ima input layer
                if (exportedInputLayer) currLayerIdx = currLayerIdx - 1;
                currLayer = (AbstractLayer)network.getLayers().get(currLayerIdx); // and get layer instance at that idx  
                
                continue;
            }
            
            if (line.startsWith("config:")) {
                continue; // skip layer config for now
            }

            if (line.startsWith("weights_shape:")) { // da li da ubacujem ovo? moram da bih mogao da pravim tenzore, mada to bi trebalo vec da zna iz konfijuracija lejera
                String shapeSubStr = line.substring(line.indexOf(":") + 1);
                String dims[] = shapeSubStr.split(",");
                // isparsiraj shape
                int shape[] = new int[dims.length];
                for (int i = 0; i < dims.length; i++) {
                    shape[i] = Integer.parseInt(dims[i].trim());
                }

                // napravi tenzor odgovarajuceg oblika oblika   
                if (shape.length == 2) {
                    weightsTensor = new Tensor2D(shape[1], shape[0]);
                } else if (shape.length == 3) {
                    weightsTensor = new Tensor3D(shape[2], shape[0], shape[1]);
                } else if (shape.length == 4) {
                    weightsTensor = new Tensor4D(shape[3], shape[2], shape[0], shape[1]); // 3, 3, 3, 64 row, cold, inCh, outCh - tako sam ih eksportovao ja | weights[kernel_height, kernel_width, kernel_depth, out_channels]
                }

                continue;
            }

            if (line.contains("weights:")) { // now reading weights section line by line
                readingWeightLines = true;
                weightsValuesBuf = new StringBuffer(); // initialize new weights buffer
                continue;
            }

            if (line.contains("biases:")) { // bias values - znaci da je zavrsio sa citanjem weights i baisao na biases, i napravi tenzor
                // pre ucitavanja biasa setuj tezine za tekuci layer
                readingWeightLines = false; // postavi zastavicu da vise ne cita tezine
                // ovde bi samo trebalo obratiti paznju da fizicki layout treba da bude column first
                // param 1 ispod je za col major, podrzumeva da je string u column major - mozda bi bilo lakse da tf exportujem u col major i ond anista ne treba da se menja ovde?
//                weightsTensor.setValuesFromStringTransposed(weightsValuesBuf.toString()); // ovde proveri da li broj vrednosti odgovara dimenzijama tenzora!!!
                if (currLayer instanceof ConvolutionalLayer) {
                    //!!!fix ovo treba upeglati da proradi import sa cuda kako treba setValuesFromStringTransposed
                    // layout column first,NCHW layout,  outch, inch, row, col      
                    // ali ovaj vec puni sa setValuesFromStringTransposed 
                    // ovo nije dobro radilo verovatno!
                    // 3, 3, 3, 64 rows, cols, inDepth, ch
                    weightsTensor.setValuesFromStringTransposed(weightsValuesBuf.toString()); // ovde proveri da li broj vrednosti odgovara dimenzijama tenzora!!!
                    ((ConvolutionalLayer) currLayer).setFilters((Tensor4D)weightsTensor);
                    ((ConvolutionalLayer) currLayer).createFilterCache();
                } else {
                    // ovo radi za 2d tensor
                    weightsTensor.setValuesFromStringTransposed(weightsValuesBuf.toString()); // ovde proveri da li broj vrednosti odgovara dimenzijama tenzora!!!                    
                    ((Tensor2D)weightsTensor).createRowsCache();
                    currLayer.setWeights(weightsTensor); // finished loading weights, zavrsi citanje weights
                }

                // ucitaj biase iz tekuce linije                
                float[] biases = parseArray(line);
                // @todo: proveri za cuda - tstiraj, izmenjeno je zbog uvodjenja Tensor1D
//                if (currLayer instanceof ConvolutionalLayer) {
//                   // currLayer.setBiases(new TensorBase(1, 1, biases.length, biases)); // ovde bi morao TensorND
//                } else {
//                    currLayer.setBiases(new Tensor1D(biases));
//                }
                
                if (currLayer instanceof ConvolutionalLayer) {
                    currLayer.setBiases(new Tensor1D(Shape.of(1, 1, biases.length), biases)); // ovde bi morao TensorND tj Tensor3D, jer nema poente da postoji ovaj konstruktor
                } else {
                    currLayer.setBiases(new Tensor1D(biases));
                }                
                continue;
            }

            if (readingWeightLines) { // nemoja da kacis sako sadrzi biases...!
                if (!line.isEmpty()) {
                    weightsValuesBuf.append(line);
                }
            }

        }
        
        br.close();
    }
    /*
    static void importWeights(FeedForwardNetwork network, String weightsFileName) throws IOException {        
        List<String> lines = Files.readAllLines(Paths.get(weightsFileName)); // How is this file created what was the training? keras_iris.py u projektu TensorFlowKerasWeightsFix 
        int layerIdx=-1;
        Tensor tensor = null;
        AbstractLayer currentLayer = null;
        StringBuffer weightsValuesBuf = new StringBuffer();
        boolean readingWeightLines = false;
        
        for(String line : lines) {
                      
            if (line.startsWith("layer:")) {
              String layerIdxStr = line.substring(line.indexOf(":")+1);
              layerIdx = Integer.parseInt(layerIdxStr); // get current layer index | u vgg imortu je  -1 jer exportovani ima input layer
              currentLayer = network.getLayers().get(layerIdx); // get layer at index to set weights for it 
              continue; // and continue to next line in file
             }
            
            if (line.startsWith("weights_shape:")) { // if line contains info about weights shape print it out eg. weights_shape:(4, 16)
                String shapeSubStr = line.substring(line.indexOf(":") + 1);
                String dims[] = shapeSubStr.split(",");
                // sve u integere i dole pravi tenzor
                int shape[] = new int[dims.length];
                for(int i=0; i<dims.length; i++) {                 // isparsiraj shape i 
                    shape[i] = Integer.parseInt(dims[i].trim());
                }
                System.out.println(shapeSubStr); // parsiraj
                tensor = new Tensor(shape[0], shape[1]); // create tensor to store weights     
                continue; // and continue to loading weight values
            }
            
            if (line.contains("weights:")) { // if the section with weights values is reached
                readingWeightLines = true; // start loading weights values for tensor
                weightsValuesBuf = new StringBuffer();
                continue;
            }
                        
            if (line.contains("biases:")) { // if the biases section is reached 
                readingWeightLines = false; // weights section has ended so stop reading weights
                String weightsValuesStr = weightsValuesBuf.toString(); //a ovaj je ""
                weightsValuesStr = weightsValuesStr.replace("[", ""); // cleanup the weights string
                weightsValuesStr = weightsValuesStr.replace("]", "");                
                tensor.setValuesFromString(weightsValuesStr); // and set loaded weights to tensor
                currentLayer.setWeights(tensor); // and set weights tensor to current layer

                // ucitaj biase iz tekuce linije                
                float[] biases = parseBiases(line); // then load biases
                currentLayer.setBiases(biases); // and set biases for current layer
                continue;
            }     
            
             
            if (readingWeightLines) { // if we're reading weights section
                weightsValuesBuf.append(line);  // then append weights:todo: use BufferedString here
            }            

        }        
    }   
    */
/*    
 static void importWeightsMnist(ConvolutionalNetwork network, String weightsFileName) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(weightsFileName));
                        
        StringBuffer weightsValuesBuf = new StringBuffer();
        boolean readingWeightLines = false;

        int currLayerIdx = -1;
        AbstractLayer currLayer = null;
        Tensor weightsTensor = null;
        
        for (String line : lines) {
            
            if (line.startsWith("layer:")) { // found next layer 
                String layerIdxStr = line.substring(line.indexOf(":") + 1); // get layer index
                currLayerIdx = Integer.parseInt(layerIdxStr);
                currLayer = network.getLayers().get(currLayerIdx); // and get layer instance at that idx
                continue;
            }

            if (line.startsWith("config:")) {
                continue; // skip layer config for now
            }

            if (line.startsWith("weights_shape:")) { // da li da ubacujem ovo? moram da bih mogao da pravim tenzore, mada to bi trebalo vec da zna iz konfijuracija lejera
                String shapeSubStr = line.substring(line.indexOf(":") + 1);
                String dims[] = shapeSubStr.split(",");
                // isparsiraj shape
                int shape[] = new int[dims.length];
                for (int i = 0; i < dims.length; i++) {
                    shape[i] = Integer.parseInt(dims[i].trim());
                }

                // napravi tenzor odgovarajuceg oblika oblika   
                if (shape.length == 2) {
                    weightsTensor = new Tensor(shape[0], shape[1]);
                } else if (shape.length == 3) {
                    weightsTensor = new Tensor(shape[0], shape[1], shape[2]);
                } else if (shape.length == 4) {
                    weightsTensor = new Tensor(shape[0], shape[1], shape[2], shape[3]);
                }

                continue;
            }

            if (line.contains("weights:")) { // now reading weights section line by line
                readingWeightLines = true;
                weightsValuesBuf = new StringBuffer(); // initialize new weights buffer
                continue;
            }

            if (line.contains("biases:")) { // bias values - znaci da je zavrsio sa citanjem weights i baisao na biases, i napravi tenzor
                // pre ucitavanja biasa setuj tezine za tekuci layer
                readingWeightLines = false;
                String weightsValuesStr2 = weightsValuesBuf.toString().replace("[", "");
                weightsValuesStr2 = weightsValuesStr2.replace("]", "");
                weightsTensor.setValuesFromString(weightsValuesStr2); // ovde proveri da li broj vrednosti odgovara dimenzijama tenzora!!!
                if (currLayer instanceof ConvolutionalLayer) {
                    ((ConvolutionalLayer) currLayer).setFilters(weightsTensor);
                } else {
                    currLayer.setWeights(weightsTensor); // FloatTensor2D, finished loading weights, zavrsi citanje weights
                }

                // ucitaj biase iz tekuce linije                
                float[] biases = parseBiases(line);
                currLayer.setBiases(biases);
                continue;
            }

            if (readingWeightLines) { // nemoja da kacis sako sadrzi biases...!
                if (!line.isEmpty()) {
                    weightsValuesBuf.append(line);
                }
            }

        }
    }    
*/

    static float[] parseArray(String line) {
        line = line.substring(line.indexOf(":") + 1);
        String[] strVals = line.split(",");
        float[] vals = new float[strVals.length];
        for (int i = 0; i < strVals.length; i++) {
            vals[i] = Float.parseFloat(strVals[i]);
        }

        return vals;
    }             

    
}
