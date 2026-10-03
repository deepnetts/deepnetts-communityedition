package deepnetts.net.train;

import javax.visrec.ml.data.DataSet;
import deepnetts.data.MLDataItem;


/**
 * Generic interface for deep learning training algorithm.
 */
public interface Trainer {
    
        /**
         * Trains this model using specified training set.
         * 
         * @param trainingSet Example data to train this model.
         */
        public void train(DataSet<? extends MLDataItem> trainingSet);
}
