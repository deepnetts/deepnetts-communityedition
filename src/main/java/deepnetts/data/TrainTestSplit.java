package deepnetts.data;

import javax.visrec.ml.data.DataSet;

/**
 * This class holds training and test data set pair.
 * Splitting data into training and test set is common practice used in machine learning, 
 * in order to test how well the machine learning model will perform with data it has not seen during the training. 
 */
public final class TrainTestSplit {
    private final DataSet<MLDataItem> trainingSet, testSet;
    // todo:store only indexes of the underlying data set, not copies of the original data set
    // add info about how it was splited is it balanced etc.
    // ideja use: int streams za indekse , mada mogu i ovako liste, ubaci batches u igru
    // todo: TrainTestVal split
    // a kako bi bilo da ubacim u dat aset metodu dataSet.trainTestSplit(0.7) // to je uobicajna praksa
    // da za minimalno koriscenje api-ja mora da zna/razume 3 koncepta/klase (max 5)
  
    public TrainTestSplit(DataSet<MLDataItem> trainingSet, DataSet<MLDataItem> testSet) {
        this.trainingSet = trainingSet;
        this.testSet = testSet;
    }

    public DataSet<MLDataItem> getTrainingSet() {
        return trainingSet;
    }
    
    public DataSet<MLDataItem> getTestSet() {
        return testSet;
    }
    
}
