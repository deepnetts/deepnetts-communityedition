package deepnetts.data;

import deepnetts.data.norm.MaxScaler;
import deepnetts.data.norm.MinMaxScaler;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.Tensor2D;
import deepnetts.tensor.Tensor3D;
import deepnetts.tensor.Tensor4D;
import deepnetts.tensor.TensorBase;
import deepnetts.util.ColumnType;
import deepnetts.util.CsvFormat;
import java.io.File;
import java.io.IOException;
import deepnetts.util.DeepNettsException;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import javax.visrec.ml.data.BasicDataSet;
import javax.visrec.ml.data.DataSet;


/**
 * Data set utility methods ans constants.
 * 
 * @author Zoran Sevarac
 */
public class DataSets {

    public final static String DELIMITER_SPACE = " ";
    public final static String DELIMITER_COMMA = ",";
    public final static String DELIMITER_SEMICOLON = ";";
    public final static String DELIMITER_TAB = "\t";

   /**
     * Creates and returns data set from specified CSV file. 
     * Empty lines are skipped
     *
     * @param csvFile CSV file
     * @param numInputs number of input values in a row
     * @param numOutputs number of output values in a row
     * @param hasColumnNames true if first row contains column names
     * @param delimiter delimiter character used to separate values in a row
     * @return instance of data set with values loaded from file
     *
     * @throws FileNotFoundException if file was not found
     * @throws IOException if there was an error reading file
     *
     * TODO: Detect if there are labels in the first line, if there are no
     * labels, set class1, class2, class3 in classifier evaluation! and detect
     * type of attributes Move this method to some factory class or something?
     * or as a default method in data set?
     *
     *  TODO: Autodetetect delimiter; column type
     *
     */
    public static TabularDataSet readCsv(File csvFile, int numInputs, int numOutputs, boolean hasColumnNames, String delimiter) throws FileNotFoundException, IOException {
        TabularDataSet dataSet = new TabularDataSet(numInputs, numOutputs);
        BufferedReader br = new BufferedReader(new FileReader(csvFile));
        String line=null;
        // auto detect column names - ako sadrzi slova onda ima imena. Sta ako su atributi nominalni? U ovoj fazi se pretpostavlja d anisu...
        // i ako u redovima ispod takodje ima stringova u istoj koloni - detect header
        if (hasColumnNames) {    // get col names from the first line
            line = br.readLine().trim();
            String[] colNames = line.split(delimiter);// fali mu jedan column name
            // todo checsk number of col names
            dataSet.setColumnNames(colNames);
        } else {
            String[] colNames = new String[numInputs+numOutputs];
            for(int i=0; i<numInputs;i++)
                colNames[i] = "in"+(i+1);

            for(int j=0; j<numOutputs;j++)
                colNames[numInputs+j] = "out"+(j+1);

            dataSet.setColumnNames(colNames);
        }

        int lineCount=0;
        while ((line = br.readLine()) != null) {
           line = line.trim();
            if (line.isEmpty()) {
                continue; // skip empty lines
            }
            lineCount++;
            
            String[] values = line.split(delimiter);
            if (values.length != (numInputs + numOutputs)) {
                throw new DeepNettsException("Wrong number of values in the row " + lineCount + ": found " + values.length + " expected " + (numInputs + numOutputs));
            }
            float[] in = new float[numInputs];
            float[] out = new float[numOutputs];

            try {
                // these methods could be extracted into parse float vectors
                for (int i = 0; i < numInputs; i++) { //parse inputs
                    if (!values[i].isEmpty() && !values[i].equals(" ") && !values[i].equalsIgnoreCase("NA")) {
                        in[i] = Float.parseFloat(values[i]);
                    } else {
                        in[i] = Float.NaN;
                    }                  
                    // restruktuiraj ovo da znas koji je prso...
                    // da li da sve bude u jednom tenzoru a da ga vadi kroz kalse??
                }

                for (int j = 0; j < numOutputs; j++) { // parse outputs
                    if (!values[j].isEmpty() && !values[j].equals(" ") && !values[j].equalsIgnoreCase("NA")) {
                        out[j] = Float.parseFloat(values[numInputs + j]);
                    } else {
                        out[j] = Float.NaN;
                    }
                }
            } catch (NumberFormatException nex) {
                throw new DeepNettsException("Error parsing csv, number expected line in " + lineCount + ": " + nex.getMessage(), nex);
                // todo: add value found
            }

            dataSet.add(new DataSetItem(in, out));
        }

        return dataSet;
    }
    
    public static TabularDataSet readCsv(String fileName, int numInputs, int numOutputs, boolean hasColumnNames, String delimiter) throws IOException {
         return readCsv(new File(fileName), numInputs, numOutputs, hasColumnNames, delimiter);
    }

    public static TabularDataSet readCsv(String fileName, int numInputs, int numOutputs, boolean hasColumnNames) throws IOException {
        return readCsv(new File(fileName), numInputs, numOutputs, hasColumnNames, ",");
    }

    public static TabularDataSet readCsv(String fileName, int numInputs, int numOutputs, String delimiter) throws IOException {
        return readCsv(new File(fileName), numInputs, numOutputs, false, delimiter);
    }

    
    
    /**
     * Create data set from CSV file, using coma (,) as default delimiter and no
     * header (column names) in first row.
     *
     * @param fileName  Name of the CSV file
     * @param numInputs Number of input columns
     * @param numOutputs Number of output columns
     * @return
     * @throws IOException
     */
    public static DataSet readCsv(String fileName, int numInputs, int numOutputs) throws IOException {
        return readCsv(new File(fileName), numInputs, numOutputs, false, ",");
    }

    // delimiter, hasHeader, column names and columnTypes
    public static CsvFormat detectCsvFormat(String fileName) throws FileNotFoundException, IOException {
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        String firstLine = br.readLine();

        // sta ako ima i navodnike ""

        // autodetect delimiter
        String delimiter = null;
        if (firstLine.contains(",")) delimiter = DELIMITER_COMMA;
        else if (firstLine.contains(";")) delimiter = DELIMITER_SEMICOLON;
        else if (firstLine.contains("\t")) delimiter = DELIMITER_TAB;
        else if (firstLine.contains(" ")) delimiter = DELIMITER_SPACE;  // da li je space delimiter za header?
        else throw new DeepNettsException("Unknown delimiter");

        boolean hasColumnNames = false;
        String[] columnNames = null;
        // da li prvi red sadrzi alfanumericka polja razdvojena delimiterima
        String[] firstLineFields = firstLine.split(delimiter);
        int colCount = firstLineFields.length;

        // mogu da budu i negativni brojevi!!!
        String intRegex = "^-?[0-9]+$"; //  "^-?(0|[1-9]\\d*)$"
        String decimalRegex = "^-?[0-9]+\\.[0-9]+$";    // "^\\d+(\\.\\d+)?$"
        String binaryRegex = "^[01]$";
        String numRegex = "^-?[0-9]+\\.?[0-9]+$";
        String alphaNumRegex = "^[a-zA-Z0-9_\\s\\-]+$"; // "^([a-zA-Z_0-9\\ \\-])+$"
        String alphaRegex = "^[a-zA-Z_\\s\\-]+$";

        boolean allNumeric = true;
        boolean allAlphaNum = true;

        // ako sadrzi sve samo numericke onda nema column names
        for(String field : firstLineFields) {
            boolean isNum = Pattern.matches(numRegex, field);
            boolean isAlphaNum = Pattern.matches(alphaNumRegex, field);
            allNumeric = allNumeric && isNum;
            allAlphaNum = allAlphaNum && isAlphaNum;
        }

        if (allNumeric) {
            hasColumnNames = false;
        } else if (allAlphaNum) { // most likely column names but might be also nominal
            columnNames = firstLineFields;
            hasColumnNames = true;
        } else { // mix of num and nominal most likely data row
            hasColumnNames = false;
        }

        // TODO: get next five rows and autodetect column types
        // int, dec, binary, string
        String[][] sampleRows=new String[5][colCount];
        for(int i=0; i<5; i++) {
            String line = br.readLine();
            String[] fields = line.split(delimiter);
            sampleRows[i] = fields; // todo trim all fields
        }

        // detect column types based on first 5 rows (or get random sample of 10 rows from first 100?)
        ColumnType colTypes[] = new ColumnType[colCount];
        for (int c=0; c<colCount; c++) {
            boolean allColsAlphaNum = true,
                    allColsBinary = true,
                    allColsDecimal = true,
                    allColsInt = true;

            for(int r=0; r<5; r++) {
                boolean isBinary = Pattern.matches(binaryRegex, sampleRows[r][c]);
                allColsBinary = allColsBinary && isBinary;

                boolean isInt = Pattern.matches(intRegex, sampleRows[r][c]);
                allColsInt = allColsInt && isInt;

                boolean isDecimal = Pattern.matches(decimalRegex, sampleRows[r][c]);
                allColsDecimal = allColsDecimal && ( isDecimal || isInt ); // moze da bud ekolona koje ima decimalne ali i int, ona se tretira kao decimalna

                boolean isAlphaNum = Pattern.matches(alphaNumRegex, sampleRows[r][c]);
                allColsAlphaNum = allColsAlphaNum && isAlphaNum;
            }

            if (allColsBinary) {
                colTypes[c] = ColumnType.BINARY;
            } else if (allColsInt) {
                colTypes[c] = ColumnType.INTEGER;
            } else if (allColsDecimal) {
                colTypes[c] = ColumnType.DECIMAL;
            } else {
                colTypes[c] = ColumnType.STRING;
            }

        }

        CsvFormat csvFormat = new CsvFormat();
        csvFormat.setDelimiter(delimiter);
        csvFormat.setColumnTypes(colTypes);
        csvFormat.setColumnNames(columnNames);
        csvFormat.setHasHeader(hasColumnNames);

        return csvFormat;

    }

    public static MaxScaler scaleToMax(DataSet dataSet) {
        MaxScaler normalizer = new MaxScaler(dataSet);
        normalizer.apply(dataSet);
        return normalizer;
    }
    
    public static MinMaxScaler scaleToMinMax(DataSet dataSet) {
        MinMaxScaler scaler = new MinMaxScaler(dataSet);
        scaler.apply(dataSet);
        return scaler;
    }    

    /**
     * Returns one hot encoded vector for the given label.
     * One-Hot encoded vector is a binary array in which each position corresponds to one label, 
     * and all elements are zero, except the one which corresponds to hotLabel which has value of one.
     * Index of hotLabel in allLabels array, determines which position in vector should be one.
     * Vector size equals to the number of labels in allLabels array.
     * 
     * @param hotLabel one label to encode
     * @param allLabels all labels (used to determine size and hot position of encoded vector)
     * 
     * @return one hot encoded vector for given label
     */
    public static float[] oneHotEncode(final String hotLabel, final String[] allLabels) {
        final float[] vect = new float[allLabels.length];

        // kako rsiti negative vektore? sve nule, a label? treba da bude sve nule, st aje label i da li ga ima u labels nizu
        // ako se hotLabel ne nalazi u allLabels bice sve nule
        
        for(int i=0; i<allLabels.length; i++) {
            if (allLabels[i].equals(hotLabel)) {
                vect[i] = 1;
            }
        }
        
        return vect;
    }

    public static TrainTestSplit trainTestSplit(DataSet<?> dataSet, double split) {
        dataSet.shuffle(); // mislim da split radi intern shuffle tako da ovo nema potrebe
        DataSet[] parts = dataSet.split(split, 1-split); // kako obezbediti da u oba data seta bude podjednaka distribucija target varijable?
        return new TrainTestSplit(parts[0], parts[1]);
    }
    
    public static TabularDataSet createBatchedDataset(TabularDataSet<?> dataSet, int batchSize) {
       Tensor1D input = (Tensor1D)dataSet.get(0).getInput();                        
       Tensor1D targetOutput = (Tensor1D)dataSet.get(0).getTargetOutput();
        
        final int numInputs = input.numElements();
        final int numOutputs = targetOutput.numElements();
        
        // jedan nacin da se smanji kolicina memorije je da svaka klas aima svoj tenzor i da pokazuju svi na isti, iskoristi hashmap za one hot tensorei klase
        TabularDataSet batchedDataSet = new TabularDataSet<>(numInputs, numOutputs);
        batchedDataSet.setColumns(dataSet.getColumns()); // ali kopija
                
        int rowIdx = 0; // dataset row idx
        int batchItemIdx = 0;
        MLDataItem batchItem = null;
        Tensor2D inBatch = null;
        Tensor2D outBatch = null;    
        
        // sta raditi ako nem adovoljno redova da se popuni poslednji batch?
        // ako su popunjeni nulama nece uciti samo ce praviti problem y abatch
        
        for(MLDataItem dsItem : dataSet.getItems()) { // broj koliko si tenzora prosao            
            input = (Tensor1D)dsItem.getInput();                        
            targetOutput = (Tensor1D)dsItem.getTargetOutput();
                                           
            if (rowIdx % batchSize == 0) { // row of the source data set
                inBatch = new Tensor2D(numInputs, batchSize); // bolje bi iblo numElements. numRows numCols
                outBatch = new Tensor2D(numOutputs, batchSize);                
                batchItem = new DataSetItem(inBatch, outBatch);
                batchItemIdx=0;
                batchedDataSet.add(batchItem);
            } 
            
           inBatch.setCol(batchItemIdx, input.getValues()); // dodaj ovu metodu prekopiraj arraj iz tenzora u ovaj segment arraycopy
           outBatch.setCol(batchItemIdx, targetOutput.getValues());
           batchItemIdx++;
            
           rowIdx++;
              
        }
        
        batchedDataSet.setColumnNames(dataSet.getColumnNames());
        
        return batchedDataSet;
    }    
    
    public static TabularDataSet createBatchedDatasetGPU(TabularDataSet<?> dataSet, int batchSize) {
       Tensor1D input = (Tensor1D)dataSet.get(0).getInput();                        
       Tensor1D targetOutput = (Tensor1D)dataSet.get(0).getTargetOutput();
        
        final int numInputs = input.numElements();
        final int numOutputs = targetOutput.numElements();
        
        // jedan nacin da se smanji kolicina memorije je da svaka klas aima svoj tenzor i da pokazuju svi na isti, iskoristi hashmap za one hot tensorei klase
        TabularDataSet batchedDataSet = new TabularDataSet<>(numInputs, numOutputs);
        batchedDataSet.setColumns(dataSet.getColumns()); // ali kopija
                
        int rowIdx = 0; // dataset row idx
        int batchItemIdx = 0;
        MLDataItem batchItem = null;
        Tensor2D inBatch = null;
        Tensor2D outBatch = null;    
        
        // sta raditi ako nem adovoljno redova da se popuni poslednji batch?
        // ako su popunjeni nulama nece uciti samo ce praviti problem y abatch
        
        for(MLDataItem dsItem : dataSet.getItems()) { // broj koliko si tenzora prosao            
            input = (Tensor1D)dsItem.getInput();                        
            targetOutput = (Tensor1D)dsItem.getTargetOutput();
                                           
            if (rowIdx % batchSize == 0) { // row of the source data set
             //   inBatch = new Tensor4D(batchSize, 1, numInputs, 1); // bolje bi iblo numElements. numRows numCols
                inBatch = new Tensor2D(numInputs, batchSize); // bolje bi iblo numElements. numRows numCols
                outBatch = new Tensor2D(numOutputs, batchSize);                
                batchItem = new DataSetItem(inBatch, outBatch);
                batchItemIdx=0;
                batchedDataSet.add(batchItem);
            } 
            
           // e sad kako prekopirati vrednosti 
         //  inBatch.setValuesAt(batchItemIdx, input.getValues());
           inBatch.setCol(batchItemIdx, input.getValues());
          // inBatch.setCol(batchItemIdx, input.getValues()); // dodaj ovu metodu prekopiraj arraj iz tenzora u ovaj segment arraycopy
          // outBatch.setCol(batchItemIdx, targetOutput.getValues());
          // outBatch.setValuesAt(batchItemIdx, targetOutput.getValues());
           outBatch.setCol(batchItemIdx, targetOutput.getValues());
           batchItemIdx++;
            
           rowIdx++;
              
        }
        
        return batchedDataSet;
    }    
    
    public static DataSet createBatchedDatasetGPU(ImageSet imageSet, int batchSize) {
       Tensor3D input = (Tensor3D)imageSet.get(0).getInput();                        
       Tensor1D targetOutput = (Tensor1D)imageSet.get(0).getTargetOutput();
        
      final int imageWidth = imageSet.getImageWidth();
      final int imageHeight = imageSet.getImageHeight();
      final int imgCh = input.depth();
      final int numOutputs = imageSet.getLabelsCount();
        
        // jedan nacin da se smanji kolicina memorije je da svaka klas aima svoj tenzor i da pokazuju svi na isti, iskoristi hashmap za one hot tensorei klase
        BasicDataSet batchedDataSet = new BasicDataSet(imageSet.getTargetColumnsNames()); // imageSet.getImageWidth(), imageSet.getImageHeight()
        batchedDataSet.setAsTargetColumns(imageSet.getTargetColumnsNames());
        
        int inputIdx = 0; // dataset row idx
        int batchItemIdx = 0;
        MLDataItem batchItem = null;
        Tensor4D inBatch = null; // napakuj slike u ovaj tenzor
        Tensor2D outBatch = null;    
        
        // sta raditi ako nem adovoljno redova da se popuni poslednji batch?
        // ako su popunjeni nulama nece uciti samo ce praviti problem y abatch
        
        batchItemIdx=0; // koji je idx u okviru mini batha batcha - kad ubaci batch size reseyuje na nulu
        for(MLDataItem dsItem : imageSet.getItems()) { // broj koliko si tenzora prosao            
            input = (Tensor3D)dsItem.getInput();                        
            targetOutput = (Tensor1D)dsItem.getTargetOutput();
                                           
            // create new mini batch item
            if (inputIdx % batchSize == 0) { // row of the source data set
                inBatch = new Tensor4D(batchSize, imgCh, imageHeight, imageWidth); // bolje bi iblo numElements. numRows numCols
                outBatch = new Tensor2D(numOutputs, batchSize);                
                batchItem = new DataSetItem(inBatch, outBatch); // ??? napravi klasu ak oje nema MlDataItem
                
                batchedDataSet.add(batchItem); //  esad kako napakovati example images u batch???
                // jedino da ih ucepam u Tensor4D
            } 
            
           // e sad kako prekopirati vrednosti 
           inBatch.setValuesAt(batchItemIdx, input.getValues());
          // inBatch.setCol(batchItemIdx, input.getValues()); // dodaj ovu metodu prekopiraj arraj iz tenzora u ovaj segment arraycopy
          // outBatch.setCol(batchItemIdx, targetOutput.getValues());
           outBatch.setCol(batchItemIdx, targetOutput.getValues());

           batchItemIdx++;            
           if (batchItemIdx == batchSize) batchItemIdx = 0;
           inputIdx++;              
        }
        
        return batchedDataSet;
    }       
        
    
    /*
     public static ImageSet createBatchedDataset(ImageSet dataSet, int batchSize) {
         // create BatchedImageSet
                 
         // idealno da ih natrpam u  ImageSet samo da posedinacni elementi budu batchevi, ako mogu da zibegnem kreiranje nove klase
         BasicDataSet batchedDataset= new BasicDataSet(); // ili nesto sto implementira dataset daj te dve pomocne kalse
         int batchIdx = 0;
         for(ExampleImage exImg : dataSet.getItems()) {
             Tensor3D input = exImg.getInput();
             TensorBase targetOut = exImg.getTargetOutput(); // tetpostavljam da je Tensor1D
             
             if (batchIdx % batchSize == 0) {
                 // create a new batch
                 Tensor4D inputBatch = new Tensor4D(batchSize, input.depth(), input.rows(), input.cols() );
                 Tensor2D targetBatch = new Tensor2D(batchSize, targetOut.numElements());
              }
             
             // create a new bacth item
             
             inputBatch.put(batchIdx, inputBatch);
             targetBatch.put(targetBatch, batchIdx);
             MLDataItem batchedDataItem = new BatchedDataItem(inputBatch, targetBatch);
             // store batch item into a current batch
             batchedDataset.add(batchedDataItem);
             // treba mi data set i data item za batched image set i nacin kako da ubacujem pojedinacne slike u batch
             batchIdx++;
             
             // store batch into a batched dataset
                          
         }
         
         return null;
     }
    */

//    public static ImageSet createImageDataSet(int imgWidth, int imgHeight, String path) { // ili jos bolje Path
//        ImageSet imageSet = new ImageSet(imgWidth, imgHeight);
//        imageSet.setResizeStrategy(ImageResize.CENTER);
//        // create lables from subdirectory names on the given pah
//        
//        // imageSet.loadLabels(new File(labelsFile));      
//        return imageSet;
//    }
    
}
