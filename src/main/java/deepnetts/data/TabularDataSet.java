package deepnetts.data;

import deepnetts.util.RandomGenerator;
import deepnetts.tensor.Tensor1D;
import deepnetts.tensor.TensorBase;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import javax.visrec.ml.data.DataSet;

/**
 * Basic data set with tabular data.
 * Used for training neural networks in Deep Netts.
 * 

 * @param <T> Type of elements in this data set.
 */
public class TabularDataSet<T extends MLDataItem> extends javax.visrec.ml.data.BasicDataSet<T> {

    private int numInputs, numOutputs; // number of inputs and outputs / target values; omoguci zadavanje razlicitih kolona za ulaze i izlaze

    protected String[] columnNames; // column names - list of so I can add or remove columns - generate names, types from csv, If not recognized, treat as string or Object

    // TODO: do we need constructor with vector dimensions annd capacity?
    
    
    protected TabularDataSet() {
        items = new ArrayList<>(); // lista elemenata koja sadrzi podatke. Da poubacujem objekte? ili nizoveprimitivne tipove. da imam columns, implementira novu verziju data seta        
    }

    /**
     * Create a new instance of BasicDataSet with specified size of input and output.
     * 
     * @param numInputs number of input features
     * @param numOutputs number of output features
     */
    public TabularDataSet(int numInputs, int numOutputs) {
        this();
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.columnNames = new String[numInputs+numOutputs];
    }

    public int getNumInputs() {
        return numInputs;
    }

    public int getNumOutputs() {
        return numOutputs;
    }

    /**
     * Split data set into specified number of part of equal sizes.
     * Utility method used during cross-validation
     * Note: this could  be default method
     *
     * @param parts
     * @return
     */
    @Override
    public DataSet[] split(int parts) {
        double partSize = (Math.round((100d / parts)))/100d;
        double[] partsArr = new double[parts];
        for (int i = 0; i < parts; i++) {
            partsArr[i] = partSize;
        }

        return split(partsArr);
    }
    
    public TrainTestSplit trainTestSplit(double splitRatio) {
        DataSet[] parts = this.split(splitRatio, 1-splitRatio); // kako obezbediti da u oba data seta bude podjednaka distribucija target varijable?
        return new TrainTestSplit(parts[0], parts[1]);        
    }

    /**
     * Splits data set into several parts specified by the input parameter
     * partSizes. Values of partSizes parameter represent the sizes of data set
     * parts that will be returned. Part sizes are decimal values that represent
     * percents, cannot be negative or zero, and their sum must be 1
     *
     * @param parts sizes of the parts in percents
     * @return parts of the data set of specified size
     */
    @Override
    public DataSet[] split(double... parts) {
        if (parts.length < 1) {
            throw new IllegalArgumentException("");
        } else if (parts.length == 1) {
            double[] newParts = new double[2];
            newParts[0] = parts[0];
            newParts[1] = 1 - parts[0];
            parts = newParts;
        }

        double partsSum = 0;
        for (int i = 0; i < parts.length; i++) {
            if (parts[i] <= 0) {
                throw new IllegalArgumentException("Value of the part cannot be zero or negative!");
            }
            partsSum += parts[i];
        }

        if (partsSum > 1) { // a sta ako je int?
            throw new IllegalArgumentException("Sum of parts cannot be larger than 1!");
        }

        DataSet[] subSets = new TabularDataSet[parts.length];
        int itemIdx = 0;

        this.shuffle(); // shuffle before splting, using global random seed - should be configurable
        for (int p = 0; p < parts.length; p++) {
            TabularDataSet subSet = new TabularDataSet(this.numInputs, this.numOutputs);
            subSet.setColumnNames(this.columnNames);
            int itemsCount = (int) (size() * parts[p]);

            for (int j = 0; j < itemsCount; j++) {
                subSet.add(items.get(itemIdx));
                itemIdx++;
            }

            subSets[p] = subSet;
        }

        return subSets;
    }

    /**
     * Shuffles the data set items using the default random generator.
     * Default rng can be initialized independently
     */
    @Override
    public void shuffle() {
        Random rnd = RandomGenerator.getDefault().getRandom();
        Collections.shuffle(items, rnd);
    }

    /**
     * Shuffles data set items using java random generator initializes with
     * specified seed
     *
     * @param seed a seed number to initialize random generator
     * @see java.util.Random
     */
    public void shuffle(int seed) {
        Random rnd = new Random(seed);
        Collections.shuffle(items, rnd);
    }

    @Override
    public String[] getColumnNames() {
        return columnNames;
    }

    @Override
    public void setColumnNames(String... columnNames) {
        this.columnNames = columnNames;
    }

    @Override
    public String[] getTargetColumnsNames() {
        String[] targetLabels = new String[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            targetLabels[i] = columnNames[numInputs + i];
        }

        return targetLabels;
    }
/*
    public boolean hasMissingValues(String colName) {
        boolean hasMissing = false;
        
        
        return hasMissing;        
    }
  */  
    float values[][];   // TODO: put values from list into this matrix
    // todo: return how many missing values it has
    public boolean hasMissingValues(int colIdx) {
        for(int rowIdx=0; rowIdx < values.length; rowIdx++) {
            if ( Float.isNaN(values[rowIdx][colIdx])) return true;
        }
        
        return false;        
    }
    
    // fo all cols
    public boolean[] hasMissingValues() {
        boolean[] hasMissing=new boolean[numInputs + numOutputs];
        
        for(int colIdx=0; colIdx < numInputs + numOutputs; colIdx++) {
            hasMissing[colIdx] = hasMissingValues(colIdx);
        }
                
        return hasMissing;        
    }    

    public int countMissingValues(int colIdx) {
        int count = 0;
        
        for(int rowIdx=0; rowIdx < values.length; rowIdx++) {
            if ( Float.isNaN(values[rowIdx][colIdx])) count++;
        }
                    
        return count;
    }
    
    public int[] countMissingValues() {
        int[] count = new int[numInputs+numOutputs];
        
        for(int colIdx=0; colIdx < numInputs+numOutputs; colIdx++) {
           count[colIdx] = countMissingValues(colIdx);
        }
                    
        return count;
    }    
    
    // desc stats for all columns: min, max, mean, median, q1, q3, iqr
    //  hasOutliers countOutliers   > q3+1.5*iqr
    
    
    
//    public DataSetStats descStat() {
//        //Tensor min, max, std, mean;
//        DataSetStats inputStats = new DataSetStats(items.get(0).getInput());
//        DataSetStats targetStats = new DataSetStats(items.get(0).getTargetOutput());
//        
//        for(MLDataItem item : items) {
//            inputStats.add(item.getInput());
//            targetStats.add(item.getTargetOutput());                        
//        }
//        
//        return inputStats;
//    }
       
}